"""Inference backends with a common interface.

    engine = create_engine("trt", weights="yolo26s_radar_fp16.engine", imgsz=1024)
    result = engine(frame_bgr)          # full-resolution screen frame
    result.boxes, result.scores, result.classes, result.semantic, result.timings

Boxes are xyxy in *full-frame* pixel coordinates; classes are dataset class ids
(1-based, see resam.config). Backends:
    trt          TensorRT engine (YOLO26/YOLO11 det or seg, end-to-end or raw head)
    ultralytics  .pt / .onnx / .engine through the ultralytics runtime
    mask2former  HuggingFace Mask2Former checkpoint (the notebook's model), torch FP16
    classical    colour-threshold + connected components, no training (baseline)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

from ..geometry import RadarROI, roi_for_image


@dataclass
class Result:
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    semantic: Optional[np.ndarray] = None      # full-frame uint8 label map (dataset ids)
    timings: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def empty():
        return Result(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32))


class Preprocessor:
    """Screen frame -> square model input, and the inverse box transform.

    crop='legacy': left HxH square + inscribed circle (what the notebook trained on)
    crop='roi'   : tight square around the measured radar ring (resam.geometry)
    """

    def __init__(self, imgsz, crop="legacy", grayscale=False, rgb=True, pad_small=False):
        self.imgsz = int(imgsz)
        # pad_small: pad a slightly smaller ROI (958 -> 960) instead of resizing. Saves ~3 ms but
        # cost 0.01-0.02 mAP50 for a model trained on resized inputs; enable for models trained so.
        self.pad_small = pad_small
        self.crop = crop
        self.grayscale = grayscale
        self.rgb = rgb
        self._roi_cache = {}
        self._mask_cache = {}

    def roi(self, frame):
        h, w = frame.shape[:2]
        if self.crop == "legacy":
            return RadarROI(h / 2.0, h / 2.0, h / 2.0, margin=1.0)
        return roi_for_image(frame, self._roi_cache)

    def __call__(self, frame):
        roi = self.roi(frame)
        img = roi.crop(frame)
        side = img.shape[0]
        if self.grayscale:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        pad = 0
        if self.pad_small and 0 < self.imgsz - side <= 32:
            # e.g. 958 px ROI -> 960 input: pad instead of resampling 10-20 px echoes (and ~3 ms)
            pad = (self.imgsz - side) // 2
            img = cv2.copyMakeBorder(img, pad, self.imgsz - side - pad, pad, self.imgsz - side - pad,
                                     cv2.BORDER_CONSTANT, value=0)
            scale = 1.0
        elif side != self.imgsz:
            img = cv2.resize(img, (self.imgsz, self.imgsz),
                             interpolation=cv2.INTER_AREA if side > self.imgsz else cv2.INTER_LINEAR)
            scale = side / float(self.imgsz)
        else:
            scale = 1.0
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x0, y0, _, _ = roi.crop_box
        return img, {"roi": roi, "offset": (x0 - pad, y0 - pad), "scale": scale, "side": side, "pad": pad}

    @staticmethod
    def boxes_to_frame(boxes, info):
        if len(boxes) == 0:
            return np.zeros((0, 4), np.float32)
        b = np.asarray(boxes, np.float32) * info["scale"]
        b[:, [0, 2]] += info["offset"][0]
        b[:, [1, 3]] += info["offset"][1]
        return b

    @staticmethod
    def semantic_to_frame(sem_crop, info, frame_shape):
        """Model-resolution label map -> full-frame label map (nearest)."""
        side = info["side"]
        pad = info.get("pad", 0)
        if pad or info["scale"] == 1.0:
            sem = sem_crop.astype(np.uint8)[pad:pad + side, pad:pad + side]
        else:
            sem = cv2.resize(sem_crop.astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST)
        out = np.zeros(frame_shape[:2], np.uint8)
        x0, y0 = info["offset"]
        h, w = frame_shape[:2]
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
        out[sy0:sy1, sx0:sx1] = sem[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
        return out


class Timer:
    def __init__(self):
        self.t = {}
        self._t0 = None
        self._k = None

    def start(self, k, sync=None):
        if sync:
            sync()
        self._k, self._t0 = k, time.perf_counter()

    def stop(self, sync=None):
        if sync:
            sync()
        self.t[self._k] = (time.perf_counter() - self._t0) * 1000.0


def create_engine(kind, **kw):
    kind = kind.lower()
    if kind == "trt":
        from .trt import TRTYoloEngine
        return TRTYoloEngine(**kw)
    if kind == "ultralytics":
        from .ultra import UltralyticsEngine
        return UltralyticsEngine(**kw)
    if kind == "mask2former":
        from .mask2former import Mask2FormerEngine
        return Mask2FormerEngine(**kw)
    if kind == "classical":
        from .classical import ClassicalEngine
        return ClassicalEngine(**kw)
    raise ValueError(f"unknown engine kind '{kind}'")
