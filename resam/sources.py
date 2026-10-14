"""Frame sources for live inference: folder replay, watched folder, HDMI capture, Ethernet
streams. All yield (frame_bgr, meta) and can be wrapped in LatestFrameReader so that the
consumer always gets the newest frame (stale frames are dropped, not queued).

URI forms accepted by make_source():
    /path/to/folder                 replay consecutive frames (sorted by name)
    folder:/path?fps=1&loop=1       same, with replay rate / looping
    watch:/mnt/radar_share          new files appearing in a folder (SMB/NFS/rsync over Ethernet)
    hdmi:/dev/video0?w=1920&h=1080&fps=30&fmt=mjpeg
                                    USB3/CSI HDMI capture card (UVC, V4L2) via GStreamer
    rtsp://host:554/stream?w=1280&h=1024
                                    IP video encoder / radar video server, HW H.264/H.265 decode
    udp://:5000?w=1280&h=1024&codec=h264
                                    RTP over UDP (e.g. gst/ffmpeg sender on the radar PC)
    http://host/video.mjpg          MJPEG over HTTP (OpenCV/FFmpeg)
    gst:<pipeline>?w=..&h=..        any GStreamer pipeline ending in raw video
    /dev/video0 | 0 | file.mp4      plain OpenCV VideoCapture
The venv's OpenCV is built without GStreamer, so GStreamer sources run gst-launch-1.0 as
a subprocess writing raw BGR frames to a pipe (NVIDIA nvv4l2decoder/nvvidconv are used).
"""

import glob
import os
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


class FolderSource:
    """Replay consecutive frames. fps=None -> as fast as the consumer pulls."""

    def __init__(self, path, fps=None, loop=False, sort="name"):
        files = [f for f in glob.glob(os.path.join(path, "*"))
                 if f.lower().endswith(IMG_EXTS) and not os.path.basename(f).startswith("._")]
        files.sort(key=os.path.getmtime if sort == "mtime" else natural_key)
        if not files:
            raise FileNotFoundError(f"no images in {path}")
        self.files, self.fps, self.loop = files, fps, loop
        self.desc = f"folder {path} ({len(files)} frames{', %.2f fps' % fps if fps else ''})"

    def __iter__(self):
        period = 1.0 / self.fps if self.fps else 0
        while True:
            for i, f in enumerate(self.files):
                t0 = time.perf_counter()
                img = cv2.imread(f, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                yield img, {"name": os.path.basename(f), "path": f, "idx": i, "t": time.time()}
                if period:
                    time.sleep(max(0.0, period - (time.perf_counter() - t0)))
            if not self.loop:
                return


class WatchFolderSource:
    """Yield files as they appear (e.g. a share the radar PC/screen grabber writes to)."""

    def __init__(self, path, poll=0.1, settle=0.05):
        self.path, self.poll, self.settle = path, poll, settle
        self.desc = f"watch {path}"

    def __iter__(self):
        seen = set(os.listdir(self.path))
        idx = 0
        while True:
            names = [n for n in os.listdir(self.path) if n not in seen and n.lower().endswith(IMG_EXTS)]
            if not names:
                time.sleep(self.poll)
                continue
            names.sort(key=natural_key)
            seen.update(names)
            time.sleep(self.settle)  # let the writer finish
            f = os.path.join(self.path, names[-1])  # newest only: real-time, not backlog
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is not None:
                yield img, {"name": names[-1], "path": f, "idx": idx, "t": time.time(), "skipped": len(names) - 1}
                idx += 1


class GstSource:
    """gst-launch-1.0 subprocess -> raw BGR frames on stdout."""

    def __init__(self, pipeline, width, height):
        if not shutil.which("gst-launch-1.0"):
            raise RuntimeError("gst-launch-1.0 not found")
        self.w, self.h = int(width), int(height)
        self.pipeline = (f"{pipeline} ! videoconvert ! video/x-raw,format=BGR,width={self.w},height={self.h} "
                         f"! fdsink fd=1 sync=false")
        self.desc = f"gstreamer {self.w}x{self.h}"

    def __iter__(self):
        cmd = ["gst-launch-1.0", "-q"] + self.pipeline.split()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        n = self.w * self.h * 3
        idx = 0
        try:
            while True:
                buf = bytearray()
                while len(buf) < n:
                    chunk = proc.stdout.read(n - len(buf))
                    if not chunk:
                        return
                    buf.extend(chunk)
                yield np.frombuffer(bytes(buf), np.uint8).reshape(self.h, self.w, 3), {
                    "name": f"frame_{idx:06d}", "idx": idx, "t": time.time()}
                idx += 1
        finally:
            proc.kill()


class OpenCVSource:
    def __init__(self, src):
        self.src = int(src) if str(src).isdigit() else src
        self.desc = f"opencv {src}"

    def __iter__(self):
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {self.src}")
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                yield frame, {"name": f"frame_{idx:06d}", "idx": idx, "t": time.time()}
                idx += 1
        finally:
            cap.release()


def _q(query, key, default=None, cast=str):
    v = query.get(key, [default])[0]
    return cast(v) if v is not None else None


def make_source(uri, fps=None, loop=False):
    if os.path.isdir(uri):
        return FolderSource(uri, fps=fps, loop=loop)
    u = urlparse(uri)
    q = parse_qs(u.query)
    w, h = _q(q, "w", 1280, int), _q(q, "h", 1024, int)
    if u.scheme == "folder":
        return FolderSource(u.path, fps=_q(q, "fps", fps, float) if q.get("fps") else fps,
                            loop=bool(_q(q, "loop", int(loop), int)), sort=_q(q, "sort", "name"))
    if u.scheme == "watch":
        return WatchFolderSource(u.path)
    if u.scheme == "hdmi":
        dev = u.path or "/dev/video0"
        cw, ch, cfps = _q(q, "w", 1920, int), _q(q, "h", 1080, int), _q(q, "fps", 30, int)
        if _q(q, "fmt", "mjpeg") == "mjpeg":
            src = (f"v4l2src device={dev} io-mode=2 ! image/jpeg,width={cw},height={ch},framerate={cfps}/1 "
                   f"! jpegdec")
        else:
            src = f"v4l2src device={dev} io-mode=2 ! video/x-raw,width={cw},height={ch},framerate={cfps}/1"
        # output at the radar screen resolution expected by the profile (w,h query, default = capture size)
        return GstSource(src, _q(q, "ow", cw, int), _q(q, "oh", ch, int))
    if u.scheme in ("rtsp", "rtsps"):
        codec = _q(q, "codec", "h264")
        loc = uri.split("?")[0]
        src = (f"rtspsrc location={loc} latency={_q(q, 'latency', 100, int)} ! rtp{codec}depay ! {codec}parse "
               f"! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx")
        return GstSource(src, w, h)
    if u.scheme == "udp":
        codec = _q(q, "codec", "h264")
        enc = "H264" if codec == "h264" else "H265"
        src = (f"udpsrc port={u.port or 5000} caps=application/x-rtp,media=video,encoding-name={enc},payload=96 "
               f"! rtp{codec}depay ! {codec}parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx")
        return GstSource(src, w, h)
    if u.scheme == "gst":
        return GstSource(uri[4:].split("?")[0], w, h)
    return OpenCVSource(uri)


class LatestFrameReader:
    """Background reader keeping only the newest frame; counts frames dropped because the
    consumer was busy. Optional change gate skips frames identical to the previous one
    (a radar screen captured at 30 fps changes only every ~1-3 s)."""

    def __init__(self, source, change_threshold=None, drop=True):
        self.source = source
        self.drop = drop  # False = backpressure (offline replay: process every frame)
        self.change_threshold = change_threshold
        self._lock = threading.Condition()
        self._item = None
        self._seq = 0
        self._done = False
        self.captured = 0
        self.dropped = 0
        self.unchanged = 0
        self._prev_small = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _changed(self, frame):
        if self.change_threshold is None:
            return True
        small = cv2.resize(frame, (160, 128), interpolation=cv2.INTER_AREA)
        prev, self._prev_small = self._prev_small, small
        if prev is None or prev.shape != small.shape:
            return True
        return float(cv2.absdiff(prev, small).mean()) > self.change_threshold

    def _run(self):
        try:
            for frame, meta in self.source:
                self.captured += 1
                if not self._changed(frame):
                    self.unchanged += 1
                    continue
                with self._lock:
                    while not self.drop and self._item is not None:
                        self._lock.wait()
                    if self._item is not None:
                        self.dropped += 1
                    self._item = (frame, meta)
                    self._seq += 1
                    self._lock.notify_all()
        finally:
            with self._lock:
                self._done = True
                self._lock.notify_all()

    def get(self, timeout=None):
        """Block until a new frame is available; None when the source is exhausted."""
        with self._lock:
            while self._item is None and not self._done:
                if not self._lock.wait(timeout):
                    return None
            item, self._item = self._item, None
            self._lock.notify_all()
            return item

    @property
    def done(self):
        return self._done and self._item is None
