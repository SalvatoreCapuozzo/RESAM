"""Annotation I/O, validation, rasterisation and dataset exporters.

Fixes relative to the notebook:
  * invalid class ids (e.g. 53 in RealImages/R_283282.json) are dropped and reported
    instead of silently producing YOLO class 52;
  * rasterisation draws large objects first, so small targets inside a Coast/Port polygon
    are not overwritten;
  * bbox-only annotations can be refined into echo-shaped pseudo-masks, instead of filled
    rectangles being used as segmentation ground truth;
  * exported filenames are made unique (folder-prefixed): 300 basenames collide across
    the dataset folders and the old YOLO builder silently overwrote them.
"""

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .config import VALID_CLASS_IDS, remap_id
from .echo import echo_mask


@dataclass
class Ann:
    cls: int
    kind: str                      # "bbox" | "segmentation"
    box: np.ndarray                # xyxy float32
    polygon: Optional[np.ndarray] = None  # (N, 2) float32

    @property
    def area(self):
        if self.polygon is not None and len(self.polygon) >= 3:
            return float(cv2.contourArea(self.polygon.astype(np.float32)))
        return float(max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1]))

    def translated(self, dx, dy):
        poly = None if self.polygon is None else self.polygon + np.array([dx, dy], np.float32)
        return Ann(self.cls, self.kind, self.box + np.array([dx, dy, dx, dy], np.float32), poly)


@dataclass
class Sample:
    base: str                      # path without extension
    image_path: str
    width: int
    height: int
    anns: List[Ann] = field(default_factory=list)
    dropped: List[dict] = field(default_factory=list)

    @property
    def uid(self):
        return unique_name(self.base)


def unique_name(base_path):
    """'IMMAGINI/Cartour delta 1.00/202109150100/R_120282' -> 'Cartour_delta_1.00__202109150100__R_120282'."""
    parts = [p for p in re.split(r"[\\/]+", base_path) if p and p not in (".", "IMMAGINI")]
    return "__".join(re.sub(r"\s+", "_", p) for p in parts[-3:])


def load_sample(base_path, image_ext=".png"):
    json_path = base_path + ".json"
    with open(json_path, "r") as f:
        data = json.load(f)
    img = data.get("image", {})
    s = Sample(base_path, base_path + image_ext, int(img.get("width", 0)), int(img.get("height", 0)))
    for a in data.get("annotations", []):
        cid = a.get("category_id")
        if cid not in VALID_CLASS_IDS:
            s.dropped.append({"reason": f"invalid class {cid}", "id": a.get("id")})
            continue
        if a.get("type") == "segmentation" and a.get("segmentation"):
            pts = np.asarray(a["segmentation"][0], np.float32).reshape(-1, 2)
            if len(pts) < 3:
                s.dropped.append({"reason": "degenerate polygon", "id": a.get("id")})
                continue
            box = np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], np.float32)
            s.anns.append(Ann(int(cid), "segmentation", box, pts))
        elif "xmin" in a:
            x0, y0 = float(a["xmin"]), float(a["ymin"])
            box = np.array([x0, y0, x0 + float(a["width"]), y0 + float(a["height"])], np.float32)
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                s.dropped.append({"reason": "degenerate box", "id": a.get("id")})
                continue
            s.anns.append(Ann(int(cid), "bbox", box))
    if s.dropped:
        warnings.warn(f"{json_path}: dropped {len(s.dropped)} annotation(s): {s.dropped[:3]}")
    return s


def list_samples(folders, image_ext=".png"):
    """All annotated base paths in the given folders (macOS '._' resource forks skipped)."""
    out = []
    for folder in folders:
        for fn in sorted(os.listdir(folder)):
            if fn.endswith(".json") and not fn.startswith("._") and fn != "categories.json":
                base = os.path.join(folder, fn[:-5])
                if os.path.exists(base + image_ext):
                    out.append(base)
    return out


# ------------------------------------------------------------------ transforms
def map_classes(anns, mapping_key):
    out = []
    for a in anns:
        c = remap_id(a.cls, mapping_key)
        if c:
            out.append(Ann(c, a.kind, a.box.copy(), None if a.polygon is None else a.polygon.copy()))
    return out


def to_roi(anns, roi, min_inside=0.3):
    """Translate annotations into ROI-crop coordinates and drop the ones mostly outside
    the radar circle (e.g. objects on the UI panel)."""
    x0, y0, x1, y1 = roi.crop_box
    side = x1 - x0
    mask = roi.local_mask((side, side))
    out = []
    for a in anns:
        t = a.translated(-x0, -y0)
        m = np.zeros((side, side), np.uint8)
        draw_ann(m, t, 1)
        tot = int(m.sum())
        if tot == 0 or (m & mask).sum() / tot < min_inside:
            continue
        t.box = np.clip(t.box, 0, side).astype(np.float32)
        if t.polygon is not None:
            t.polygon = np.clip(t.polygon, 0, side).astype(np.float32)
        out.append(t)
    return out


# ------------------------------------------------------------------ rasterisation
def draw_ann(canvas, a, value):
    if a.polygon is not None:
        cv2.fillPoly(canvas, [np.round(a.polygon).astype(np.int32)], int(value))
    else:
        x0, y0, x1, y1 = np.round(a.box).astype(int)
        canvas[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = value


def rasterize_semantic(anns, shape, ignore_overlap_order=False):
    """Semantic label map. Largest annotations are drawn first so small targets survive."""
    m = np.zeros(shape[:2], np.uint8)
    order = anns if ignore_overlap_order else sorted(anns, key=lambda a: -a.area)
    for a in order:
        draw_ann(m, a, a.cls)
    return m


def rasterize_instances(anns, shape):
    """Instance id map (1..N) + {instance_id: class}. Largest first, as above."""
    m = np.zeros(shape[:2], np.int32)
    id2cls = {}
    for i, a in enumerate(sorted(anns, key=lambda a: -a.area), start=1):
        draw_ann(m, a, i)
        id2cls[i] = a.cls
    present = set(np.unique(m).tolist())
    return m, {k: v for k, v in id2cls.items() if k in present}


def refine_boxes_with_echo(image_bgr, anns, classes=(1, 2, 3, 6), min_fill=0.05, dilate=1):
    """Turn bbox annotations of echo-like classes into echo-shaped polygons (pseudo-masks):
    inside each box, keep the radar-echo-coloured pixels. Falls back to the box when the
    box contains almost no echo pixels. Returns a new list."""
    em = echo_mask(image_bgr)
    if dilate:
        em = cv2.dilate(em, np.ones((3, 3), np.uint8), iterations=dilate)
    out = []
    for a in anns:
        if a.kind != "bbox" or a.cls not in classes:
            out.append(a)
            continue
        x0, y0, x1, y1 = np.round(a.box).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        sub = em[y0:y1, x0:x1]
        if sub.size == 0 or sub.mean() / 255.0 < min_fill:
            out.append(a)
            continue
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= 2]
        if not cnts:
            out.append(a)
            continue
        hull = cv2.convexHull(np.concatenate(cnts)).reshape(-1, 2).astype(np.float32) + [x0, y0]
        if len(hull) < 3:
            out.append(a)
            continue
        out.append(Ann(a.cls, "segmentation", a.box.copy(), hull))
    return out


# ------------------------------------------------------------------ exporters
def yolo_det_lines(anns, img_w, img_h, class_offset=1):
    lines = []
    for a in anns:
        x0, y0, x1, y1 = a.box
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        cx, cy = (x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h
        lines.append(f"{a.cls - class_offset} {cx:.6f} {cy:.6f} {(x1 - x0) / img_w:.6f} {(y1 - y0) / img_h:.6f}")
    return lines


def yolo_seg_lines(anns, img_w, img_h, class_offset=1):
    lines = []
    for a in anns:
        poly = a.polygon
        if poly is None:
            x0, y0, x1, y1 = a.box
            poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        p = poly / np.array([img_w, img_h], np.float32)
        p = np.clip(p, 0, 1)
        lines.append(f"{a.cls - class_offset} " + " ".join(f"{v:.6f}" for v in p.reshape(-1)))
    return lines
