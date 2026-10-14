#!/usr/bin/env python3
"""Real-time RESAM dashboard: live radar frames -> inference -> tracking -> browser.

Sources (see resam/sources.py): a folder of consecutive frames, a watched network folder,
an HDMI capture card, or an Ethernet video stream. Open http://<jetson-ip>:8080 from any
machine on the network; add --show to also draw on the Jetson's own HDMI monitor.

  # replay a recorded sequence at the radar's sweep rate, with live accuracy vs labels
  python tools/dashboard.py --source "IMMAGINI/Cartour delta 1.30/202109150130" --fps 2 --loop \
      --engine trt:models/yolo26s_roi960_fp16.engine@crop=roi

  # HDMI capture card (radar console output) / RTSP encoder on the ship LAN
  python tools/dashboard.py --source "hdmi:/dev/video0?w=1280&h=1024&fps=30" --change-threshold 0.5 ...
  python tools/dashboard.py --source "rtsp://192.168.1.50:554/radar?w=1280&h=1024" ...

  # no model yet: classical echo baseline
  python tools/dashboard.py --source <folder> --engine classical

Endpoints: /  (dashboard) · /stream.mjpg · /snapshot.jpg · /stats.json · /healthz
When a frame comes from a folder and <frame>.json labels exist next to it, the dashboard
also shows running precision / Pd / false alarms per frame against those labels.
"""

import argparse
import collections
import csv
import json
import os
import sys
import threading
import time
import traceback
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import parse_spec  # noqa: E402
from resam.annotations import load_sample, map_classes  # noqa: E402
from resam.config import CLASSES, MAPPINGS  # noqa: E402
from resam.engines import create_engine  # noqa: E402
from resam.geometry import roi_for_image  # noqa: E402
from resam.metrics import DetectionEvaluator, LatencyStats  # noqa: E402
from resam.sources import LatestFrameReader, make_source  # noqa: E402
from resam.telemetry import Telemetry  # noqa: E402
from resam.tracking import SweepTracker  # noqa: E402
from resam.viz import CLASS_COLORS, render  # noqa: E402

HIST = 180


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.jpeg_seq = 0
        self.cond = threading.Condition(self.lock)
        self.stats = {}
        self.hist = {k: collections.deque(maxlen=HIST) for k in ("t", "total", "infer", "power_w", "gpu", "fps", "n_det")}
        self.error = None


def put_jpeg(state, img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        with state.cond:
            state.jpeg = buf.tobytes()
            state.jpeg_seq += 1
            state.cond.notify_all()


def worker(args, state, tel):
    try:
        kind, kw = parse_spec(args.engine)
        eng = create_engine(kind, **kw)
        names = dict(CLASSES)
        names.update(getattr(eng, "class_names", {}) or {})
        src = make_source(args.source, fps=args.fps, loop=args.loop)
        reader = LatestFrameReader(src, change_threshold=args.change_threshold, drop=not args.no_drop)
        tracker = SweepTracker() if not args.no_track else None
        lat = LatencyStats(window=300)
        live = DetectionEvaluator(MAPPINGS[args.mapping]["classes"])
        roi_cache = {}
        log = None
        if args.record:
            os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
            log_f = open(args.record, "w", newline="", buffering=1)  # line-buffered: survives kill
            log = csv.writer(log_f)
            log.writerow(["t", "frame", "pre_ms", "infer_ms", "post_ms", "total_ms", "n_det", "n_tracks",
                          "power_mw", "gpu_pct"])
        n, t_prev, fps_ema = 0, None, None
        live_metrics, class_counts = None, {}
        with state.lock:
            state.stats["source"] = getattr(src, "desc", args.source)
            state.stats["engine"] = args.engine
        while True:
            item = reader.get(timeout=1.0)
            if item is None:
                if reader.done:
                    with state.lock:
                        state.stats["status"] = "source ended"
                    if not args.loop:
                        break
                continue
            frame, meta = item
            t0 = time.perf_counter()
            r = eng(frame)
            t_inf = (time.perf_counter() - t0) * 1000
            tracks = []
            if tracker is not None:
                t1 = time.perf_counter()
                tracks = tracker.update(r.boxes, r.scores, r.classes)
                r.timings["track"] = (time.perf_counter() - t1) * 1000
            roi = roi_for_image(frame, roi_cache)

            gt = None
            p = meta.get("path")
            if p and os.path.exists(os.path.splitext(p)[0] + ".json"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s = load_sample(os.path.splitext(p)[0], os.path.splitext(p)[1])
                g = [a for a in map_classes(s.anns, args.mapping)
                     if np.hypot((a.box[0] + a.box[2]) / 2 - roi.cx, (a.box[1] + a.box[3]) / 2 - roi.cy) <= roi.r]
                gb = np.array([a.box for a in g], np.float32).reshape(-1, 4)
                gc = np.array([a.cls for a in g], np.int32)
                lut = MAPPINGS[args.mapping]["lut"]
                pc = np.array([lut.get(int(c), 0) for c in r.classes], np.int32)
                k = pc > 0
                live.add(gb, gc, r.boxes[k], r.scores[k], pc[k])
                gt = (gb, gc)
                if n % 5 == 0:
                    live_metrics = {"iou": live.operating_point(args.conf, "iou", 0.5)["overall"],
                                    "center": live.operating_point(args.conf, "center")["overall"],
                                    "frames": len(live.images)}

            t_r = time.perf_counter()
            vis = render(frame, r, names, roi=roi, tracks=tracks, gt=gt if args.show_gt else None)
            if args.max_width and vis.shape[1] > args.max_width:
                s_ = args.max_width / vis.shape[1]
                vis = cv2.resize(vis, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA)
            put_jpeg(state, vis, args.jpeg_quality)
            if args.show:
                cv2.imshow("RESAM", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            t_render = (time.perf_counter() - t_r) * 1000

            now = time.time()
            if t_prev is not None:
                inst = 1.0 / max(now - t_prev, 1e-6)
                fps_ema = inst if fps_ema is None else 0.8 * fps_ema + 0.2 * inst
            t_prev = now
            lat.add(total=t_inf, render=t_render, **r.timings)
            class_counts = {}
            for c in r.classes:
                class_counts[names.get(int(c), str(int(c)))] = class_counts.get(names.get(int(c), str(int(c))), 0) + 1
            snap = tel.snapshot()
            n += 1
            with state.lock:
                h = state.hist
                h["t"].append(now)
                h["total"].append(round(t_inf, 2))
                h["infer"].append(round(r.timings.get("infer", 0), 2))
                h["power_w"].append(round((snap.get("power_mw") or 0) / 1000, 2))
                h["gpu"].append(snap.get("gpu_pct"))
                h["fps"].append(round(fps_ema or 0, 2))
                h["n_det"].append(int(len(r.boxes)))
                state.stats.update({
                    "status": "running", "frame": meta.get("name"), "frames_processed": n,
                    "captured": reader.captured, "dropped": reader.dropped, "unchanged_skipped": reader.unchanged,
                    "fps": fps_ema, "latency": lat.summary(), "last_timings": r.timings,
                    "n_det": int(len(r.boxes)), "class_counts": class_counts,
                    "tracks": {"active": len(tracks), "confirmed": len(tracker.confirmed()) if tracker else None,
                               "noise_like": len(tracker.noise_like()) if tracker else None,
                               **(tracker.stats if tracker else {})},
                    "telemetry": snap, "live_metrics": live_metrics, "mapping": args.mapping,
                    "conf": args.conf, "class_colors": {names.get(k, str(k)): "#%02x%02x%02x" % v[::-1]
                                                        for k, v in CLASS_COLORS.items()},
                })
            if log:
                log.writerow([f"{now:.3f}", meta.get("name"), *[f"{r.timings.get(k, 0):.2f}" for k in ("pre", "infer", "post")],
                              f"{t_inf:.2f}", len(r.boxes), len(tracks), snap.get("power_mw"), snap.get("gpu_pct")])
    except Exception as e:
        traceback.print_exc()
        with state.lock:
            state.error = f"{type(e).__name__}: {e}"
            state.stats["status"] = "error"


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif self.path.startswith("/stats.json"):
                with state.lock:
                    d = dict(state.stats)
                    d["history"] = {k: list(v) for k, v in state.hist.items()}
                    d["error"] = state.error
                self._send(200, "application/json", json.dumps(d, default=lambda o: None).encode())
            elif self.path.startswith("/snapshot.jpg"):
                with state.lock:
                    j = state.jpeg
                self._send(200 if j else 503, "image/jpeg", j or b"")
            elif self.path.startswith("/healthz"):
                self._send(200, "text/plain", (state.stats.get("status", "starting")).encode())
            elif self.path.startswith("/stream.mjpg"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        with state.cond:
                            state.cond.wait_for(lambda: state.jpeg_seq != seq, timeout=5)
                            j, seq = state.jpeg, state.jpeg_seq
                        if j is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                         str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self._send(404, "text/plain", b"not found")
    return H


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RESAM Live</title>
<style>
:root{color-scheme:dark;--bg:#111110;--surface:#1a1a19;--line:#2c2c2a;--text:#f0efec;--text2:#c3c2b7;--muted:#8d8c84;
--series:#3987e5;--good:#3fa35a;--warn:#c98500;--crit:#e66767}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{padding:16px;max-width:1700px;margin:0 auto}
header{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:baseline;margin-bottom:12px}
h1{font-size:18px;margin:0;font-weight:600}.meta{color:var(--text2);font-size:13px}
.pill{display:inline-flex;gap:6px;align-items:center;padding:2px 10px;border-radius:999px;background:var(--surface);border:1px solid var(--line);font-size:12px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted)}
.grid{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);gap:16px}
@media (max-width:1000px){.grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:12px;font-weight:600;color:var(--text2);margin:0 0 8px;text-transform:uppercase;letter-spacing:.04em}
#live{width:100%;height:auto;display:block;border-radius:6px;background:#000}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px}
.tile{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.tile .k{font-size:11px;color:var(--muted)}.tile .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .s{font-size:11px;color:var(--text2);font-variant-numeric:tabular-nums}
.charts{display:grid;grid-template-columns:1fr;gap:10px;margin-top:10px}
.chart{position:relative}.chart canvas{width:100%;height:90px;display:block}
.chart .lbl{display:flex;justify-content:space-between;font-size:12px;color:var(--text2);margin-bottom:2px}
.tip{position:absolute;pointer-events:none;background:#000c;border:1px solid var(--line);color:var(--text);font-size:12px;padding:2px 6px;border-radius:4px;display:none;white-space:nowrap}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:3px 6px;text-align:left;border-bottom:1px solid var(--line);font-size:13px}
th{color:var(--muted);font-weight:500}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.err{color:var(--crit);white-space:pre-wrap}
</style></head><body><div class="wrap">
<header><h1>RESAM · live radar inference</h1>
<span class="pill"><span class="dot" id="sdot"></span><span id="status">connecting…</span></span>
<span class="meta" id="src"></span><span class="meta" id="eng"></span></header>
<div class="grid">
 <div class="card"><h2>Live frame <span class="meta" id="fname" style="text-transform:none;letter-spacing:0"></span></h2>
  <img id="live" src="/stream.mjpg" alt="Annotated radar frame"></div>
 <div>
  <div class="card"><h2>Runtime</h2><div class="tiles">
   <div class="tile"><div class="k">Throughput</div><div class="v" id="fps">–</div><div class="s">frames / s</div></div>
   <div class="tile"><div class="k">Latency p50</div><div class="v" id="p50">–</div><div class="s" id="p95">p95 –</div></div>
   <div class="tile"><div class="k">Model infer p50</div><div class="v" id="inf">–</div><div class="s" id="prepost">pre – / post –</div></div>
   <div class="tile"><div class="k">Board power</div><div class="v" id="pw">–</div><div class="s" id="energy">– mJ / frame</div></div>
   <div class="tile"><div class="k">GPU load</div><div class="v" id="gpu">–</div><div class="s" id="temp">– °C</div></div>
   <div class="tile"><div class="k">RAM</div><div class="v" id="ram">–</div><div class="s" id="ramt">of – GB</div></div>
   <div class="tile"><div class="k">Frames</div><div class="v" id="nproc">–</div><div class="s" id="drop">dropped – · unchanged –</div></div>
  </div>
  <div class="charts">
   <div class="chart"><div class="lbl"><span>End-to-end latency (ms)</span><span id="c1v"></span></div><canvas id="c1"></canvas><div class="tip" id="t1"></div></div>
   <div class="chart"><div class="lbl"><span>Board power, VDD_IN (W)</span><span id="c2v"></span></div><canvas id="c2"></canvas><div class="tip" id="t2"></div></div>
   <div class="chart"><div class="lbl"><span>Detections per frame</span><span id="c3v"></span></div><canvas id="c3"></canvas><div class="tip" id="t3"></div></div>
  </div></div>
  <div class="card" style="margin-top:16px"><h2>Detections & tracks</h2>
   <table><thead><tr><th>Class</th><th>Now</th></tr></thead><tbody id="cls"></tbody></table>
   <div class="tiles" style="margin-top:8px">
    <div class="tile"><div class="k">Active tracks</div><div class="v" id="tra">–</div></div>
    <div class="tile"><div class="k">Confirmed (≥3 sweeps)</div><div class="v" id="trc">–</div></div>
    <div class="tile"><div class="k">Flicker (clutter-like)</div><div class="v" id="trn">–</div></div>
   </div></div>
  <div class="card" style="margin-top:16px"><h2>Live accuracy vs labels <span class="meta" id="lmm" style="text-transform:none;letter-spacing:0"></span></h2>
   <div id="nolabels" class="meta">No label files next to the incoming frames.</div>
   <div class="tiles" id="lm" hidden>
    <div class="tile"><div class="k">Precision (IoU≥.5)</div><div class="v" id="lp">–</div></div>
    <div class="tile"><div class="k">Pd / recall (IoU≥.5)</div><div class="v" id="lr">–</div></div>
    <div class="tile"><div class="k">Pd centre-hit</div><div class="v" id="lpc">–</div></div>
    <div class="tile"><div class="k">False alarms / frame</div><div class="v" id="lfa">–</div></div>
   </div></div>
  <div class="err" id="err"></div>
 </div></div></div>
<script>
const $=id=>document.getElementById(id);
const f=(v,d=1)=>v==null||isNaN(v)?'–':Number(v).toFixed(d);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function spark(cid,tid,vals,unit,d){
  const c=$(cid),dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;
  const g=c.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,w,h);
  const v=vals.map(x=>x==null?NaN:+x);const ok=v.filter(x=>!isNaN(x));if(ok.length<2)return;
  const lo=0,hi=Math.max(...ok)*1.15||1,X=i=>4+i*(w-8)/(v.length-1),Y=y=>h-6-(y-lo)/(hi-lo)*(h-14);
  g.strokeStyle=css('--line');g.lineWidth=1;g.beginPath();g.moveTo(0,Y(0)+.5);g.lineTo(w,Y(0)+.5);g.stroke();
  g.fillStyle=css('--muted');g.font='11px system-ui';g.fillText(f(hi/1.15,d)+' '+unit,4,11);
  g.strokeStyle=css('--series');g.lineWidth=2;g.lineJoin='round';g.beginPath();let m=false;
  v.forEach((y,i)=>{if(isNaN(y)){m=false;return}m?g.lineTo(X(i),Y(y)):g.moveTo(X(i),Y(y));m=true});g.stroke();
  c._v=v;c._X=X;c._Y=Y;c._u=unit;c._d=d;c._t=tid;
}
for(const id of ['c1','c2','c3']){const c=$(id);
 c.addEventListener('mousemove',e=>{if(!c._v)return;const r=c.getBoundingClientRect(),x=e.clientX-r.left;
  const i=Math.max(0,Math.min(c._v.length-1,Math.round((x-4)/((r.width-8)/(c._v.length-1)))));const t=$(c._t),y=c._v[i];
  if(isNaN(y)){t.style.display='none';return}t.textContent=f(y,c._d)+' '+c._u+' · '+(c._v.length-1-i)+' frames ago';
  t.style.display='block';t.style.left=Math.min(r.width-120,Math.max(0,c._X(i)-40))+'px';t.style.top=(c._Y(y)+6)+'px';
  const g=c.getContext('2d');spark(id,c._t,c._v,c._u,c._d);g.strokeStyle=css('--text2');g.lineWidth=1;g.beginPath();g.moveTo(c._X(i),0);g.lineTo(c._X(i),r.height);g.stroke();
  g.fillStyle=css('--series');g.strokeStyle=css('--surface');g.lineWidth=2;g.beginPath();g.arc(c._X(i),c._Y(y),4,0,7);g.fill();g.stroke();});
 c.addEventListener('mouseleave',()=>{$(c._t||'t1').style.display='none'});}
async function tick(){
 try{const s=await (await fetch('/stats.json',{cache:'no-store'})).json();
  const st=s.status||'starting';$('status').textContent=st;$('sdot').style.background=st=='running'?css('--good'):st=='error'?css('--crit'):css('--warn');
  $('src').textContent=s.source||'';$('eng').textContent=s.engine?('engine: '+s.engine):'';$('fname').textContent=s.frame||'';
  const L=s.latency||{},T=L.total||{},I=L.infer||{},P=L.pre||{},Q=L.post||{},tel=s.telemetry||{};
  $('fps').textContent=f(s.fps);$('p50').textContent=f(T.p50)+' ms';$('p95').textContent='p95 '+f(T.p95)+' · p99 '+f(T.p99);
  $('inf').textContent=f(I.p50)+' ms';$('prepost').textContent='pre '+f(P.p50)+' / post '+f(Q.p50)+' ms';
  $('pw').textContent=tel.power_mw?f(tel.power_mw/1000,2)+' W':'n/a';
  $('energy').textContent=(tel.power_mw&&s.fps)?f(tel.power_mw/s.fps,0)+' mJ / frame':'telemetry: '+(tel.backend||'–');
  $('gpu').textContent=tel.gpu_pct!=null?f(tel.gpu_pct,0)+'%':'–';$('temp').textContent='GPU '+f(tel.temp_gpu)+' °C';
  $('ram').textContent=tel.ram_used_mb?f(tel.ram_used_mb/1024,2)+' GB':'–';$('ramt').textContent='of '+f((tel.ram_total_mb||0)/1024,1)+' GB';
  $('nproc').textContent=s.frames_processed??'–';$('drop').textContent='dropped '+(s.dropped??'–')+' · unchanged '+(s.unchanged_skipped??'–');
  const H=s.history||{};spark('c1','t1',H.total||[],'ms',1);spark('c2','t2',H.power_w||[],'W',2);spark('c3','t3',H.n_det||[],'',0);
  $('c1v').textContent=f((H.total||[]).slice(-1)[0])+' ms';$('c2v').textContent=f((H.power_w||[]).slice(-1)[0],2)+' W';$('c3v').textContent=(H.n_det||[]).slice(-1)[0]??'';
  const cc=s.class_counts||{},col=s.class_colors||{};$('cls').innerHTML=Object.keys(cc).sort().map(k=>`<tr><td><span class="sw" style="background:${col[k]||'#999'}"></span>${k}</td><td>${cc[k]}</td></tr>`).join('')||'<tr><td colspan=2 class="meta">none</td></tr>';
  const tr=s.tracks||{};$('tra').textContent=tr.active??'–';$('trc').textContent=tr.confirmed??'–';$('trn').textContent=tr.noise_like??'–';
  const lm=s.live_metrics;$('lm').hidden=!lm;$('nolabels').hidden=!!lm;
  if(lm){$('lp').textContent=f(lm.iou.P,3);$('lr').textContent=f(lm.iou.R,3);$('lpc').textContent=f(lm.center.Pd,3);$('lfa').textContent=f(lm.iou.FA_per_frame,2);
   $('lmm').textContent=`${lm.frames} labelled frames · mapping ${s.mapping} · conf ≥ ${s.conf}`;}
  $('err').textContent=s.error||'';
 }catch(e){$('status').textContent='disconnected';$('sdot').style.background=css('--crit')}
}
setInterval(tick,1000);tick();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--engine", default="classical", help="spec as in tools/benchmark.py")
    ap.add_argument("--fps", type=float, default=None, help="folder replay rate (default: as fast as possible)")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-drop", action="store_true", help="process every frame (offline replay)")
    ap.add_argument("--change-threshold", type=float, default=None,
                    help="skip frames whose mean abs diff vs previous is below this (HDMI at 30 fps)")
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--mapping", default="original", choices=list(MAPPINGS))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--show-gt", action="store_true", help="draw label boxes (white) when available")
    ap.add_argument("--show", action="store_true", help="also display on the Jetson's monitor (OpenCV window)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--record", default=None, help="CSV log of per-frame timings/power")
    args = ap.parse_args()

    state = State()
    tel = Telemetry(interval=0.5)
    threading.Thread(target=worker, args=(args, state, tel), daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    srv.daemon_threads = True
    print(f"RESAM dashboard on http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
