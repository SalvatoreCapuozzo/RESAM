"""Training-free baseline: echo colour threshold + connected components."""

from ..echo import ClassicalDetector
from ..geometry import roi_for_image
from . import Result, Timer


class ClassicalEngine:
    name = "classical"

    def __init__(self, land_area_frac=0.004, min_area=6, own_ship_xy=None, **_):
        self.det = ClassicalDetector(land_area_frac=land_area_frac, min_area=min_area, own_ship_xy=own_ship_xy)
        self._roi_cache = {}
        self._mask_cache = {}
        self.class_names = {1: "Ship", 4: "Coast/Port", 5: "My Ship"}

    def __call__(self, frame):
        t = Timer()
        t.start("pre")
        key = frame.shape[:2]
        if key not in self._mask_cache:
            self._mask_cache[key] = roi_for_image(frame, self._roi_cache).full_mask(frame.shape) * 255
        t.stop()
        t.start("infer")
        boxes, scores, classes, _, _ = self.det(frame, self._mask_cache[key])
        t.stop()
        t.t["post"] = 0.0
        r = Result(boxes, scores, classes)
        r.timings = t.t
        return r
