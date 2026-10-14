"""Radar-screen geometry: display-circle ROI, crop/mask, pixel <-> geographic conversion.

The original notebook assumed the radar video is the circle inscribed in the left HxH
square. Measured on the dataset this is off by 20-35 px (and includes the bearing-scale
numerals as "echoes"), so the ROI is taken from a measured display profile, or fitted
from the static ring of a few frames when the display is unknown.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import profile_for_size

NM_EARTH_RADIUS = 3440.065


@dataclass
class RadarROI:
    cx: float
    cy: float
    r: float
    margin: float = 0.985  # shrink the mask a little to drop the ring/tick pixels

    @property
    def crop_box(self):
        """Square (x0, y0, x1, y1) tightly enclosing the radar circle."""
        x0 = int(round(self.cx - self.r))
        y0 = int(round(self.cy - self.r))
        side = int(round(2 * self.r))
        return x0, y0, x0 + side, y0 + side

    def crop(self, image, fill=0):
        """Crop the square around the circle (zero-padded where it leaves the image)
        and black out everything outside the circle."""
        x0, y0, x1, y1 = self.crop_box
        h, w = image.shape[:2]
        out_shape = (y1 - y0, x1 - x0) + image.shape[2:]
        out = np.full(out_shape, fill, dtype=image.dtype)
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
        m = self._cached_mask(out.shape[:2])
        if fill == 0:
            return cv2.bitwise_and(out, out, mask=m)  # ~15x faster than boolean indexing
        out[m == 0] = fill
        return out

    def _cached_mask(self, shape):
        cache = self.__dict__.setdefault("_mask_cache", {})
        if shape not in cache:
            cache[shape] = self.local_mask(shape)
        return cache[shape]

    def local_mask(self, shape):
        """Circle mask in crop coordinates."""
        m = np.zeros(shape[:2], dtype=np.uint8)
        c = (shape[1] // 2, shape[0] // 2)
        cv2.circle(m, c, int(self.r * self.margin), 1, thickness=-1)
        return m

    def full_mask(self, shape):
        m = np.zeros(shape[:2], dtype=np.uint8)
        cv2.circle(m, (int(self.cx), int(self.cy)), int(self.r * self.margin), 1, thickness=-1)
        return m

    def to_crop(self, xy):
        x0, y0, _, _ = self.crop_box
        xy = np.asarray(xy, dtype=np.float32)
        return xy - np.array([x0, y0], dtype=np.float32)

    def from_crop(self, xy):
        x0, y0, _, _ = self.crop_box
        xy = np.asarray(xy, dtype=np.float32)
        return xy + np.array([x0, y0], dtype=np.float32)


def fit_display_circle(frames, r_range=(0.38, 0.52), edge_persistence=0.6):
    """Fit the radar video ring from a list of BGR frames of the same display.

    Echoes change sweep to sweep while the bearing ring is static, so we keep only edges
    present in >= `edge_persistence` of the frames and vote for (cx, cy, r) restricted to
    the left HxH area. Coarse search on a downscaled map, then a full-res refinement.
    """
    if not frames:
        raise ValueError("need at least one frame")
    h, w = frames[0].shape[:2]
    acc = np.zeros((h, w), np.float32)
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        acc += cv2.Canny(g, 60, 150) > 0
    static = acc / len(frames) >= (edge_persistence if len(frames) > 2 else 0.5)
    static[:, min(w, h + h // 8):] = False  # the right side is the UI panel

    def vote(ys, xs, centers, rmin, rmax):
        best = (-1, 0, 0, 0)
        for cx, cy in centers:
            d = np.hypot(xs - cx, ys - cy).astype(np.int32)
            hist = np.bincount(d, minlength=rmax + 2)[rmin:rmax]
            if hist.size == 0:
                continue
            sc = np.convolve(hist, np.ones(3), "same")
            i = int(sc.argmax())
            if sc[i] > best[0]:
                best = (sc[i], cx, cy, rmin + i)
        return best

    s = 256.0 / h
    small = cv2.resize(static.astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(small)
    grid = [(x, y) for y in range(int(256 * 0.35), int(256 * 0.65), 2)
            for x in range(int(256 * 0.35), int(256 * 0.65), 2)]
    _, cx, cy, _ = vote(ys, xs, grid, int(256 * r_range[0]), int(256 * r_range[1]))
    cx, cy = cx / s, cy / s
    ys, xs = np.nonzero(static)
    fine = [(x, y) for y in range(int(cy - 6), int(cy + 7)) for x in range(int(cx - 6), int(cx + 7))]
    _, cx, cy, r = vote(ys, xs, fine, int(h * r_range[0]), int(h * r_range[1]))
    return RadarROI(float(cx), float(cy), float(r))


def roi_for_image(image, cache=None):
    """ROI from a known display profile, else inscribed-square fallback (legacy behaviour).
    For unknown displays prefer fit_display_circle() on a handful of frames and cache it."""
    h, w = image.shape[:2]
    if cache is not None and (w, h) in cache:
        return cache[(w, h)]
    _, prof = profile_for_size(w, h)
    roi = RadarROI(*prof["circle"]) if prof else RadarROI(h / 2.0, h / 2.0, h / 2.0)
    if cache is not None:
        cache[(w, h)] = roi
    return roi


# ---------------------------------------------------------------- geo-referencing
def pixel_to_latlon(px, py, own_px, own_py, pixels_per_nm, heading_deg, own_lat, own_lon):
    """Great-circle projection of a screen pixel (head-up display) to lat/lon.
    Same math as radar_labeler.calculate_target_lat_lon, without the hardcoded origin."""
    dx = px - own_px
    dy = own_py - py
    dist_nm = math.hypot(dx, dy) / pixels_per_nm
    if dist_nm == 0:
        return own_lat, own_lon
    brng = math.radians((math.degrees(math.atan2(dx, dy)) + heading_deg) % 360)
    d = dist_nm / NM_EARTH_RADIUS
    lat1, lon1 = math.radians(own_lat), math.radians(own_lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d) * math.cos(lat1),
                             math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def latlon_to_pixel(lat, lon, own_px, own_py, pixels_per_nm, heading_deg, own_lat, own_lon):
    """Inverse of pixel_to_latlon (haversine range/bearing -> screen pixel)."""
    phi1, phi2 = math.radians(own_lat), math.radians(lat)
    dphi, dlmb = math.radians(lat - own_lat), math.radians(lon - own_lon)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    dist_nm = 2 * NM_EARTH_RADIUS * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    y = math.sin(dlmb) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlmb)
    rel = math.radians((math.degrees(math.atan2(y, x)) - heading_deg) % 360)
    d = dist_nm * pixels_per_nm
    return own_px + d * math.sin(rel), own_py - d * math.cos(rel)


def pixels_per_nm(roi, range_nm):
    """In centred mode the video ring radius equals the selected range."""
    return roi.r / float(range_nm)
