"""Ultralytics runtime backend (.pt / .onnx / .engine). Convenient, not the fastest path:
prefer resam.engines.trt for deployment."""

import numpy as np

from . import Preprocessor, Result, Timer


class UltralyticsEngine:
    name = "ultralytics"

    def __init__(self, weights, imgsz=1024, crop="legacy", conf=0.25, half=True, class_ids=None, **_):
        from ultralytics import YOLO
        self.model = YOLO(weights, task=None)
        self.pre = Preprocessor(imgsz, crop=crop, rgb=False)  # ultralytics expects BGR numpy
        self.conf = conf
        self.half = half
        names = self.model.names if hasattr(self.model, "names") else {}
        self.class_ids = class_ids or {k: k + 1 for k in names}
        self.class_names = {self.class_ids[k]: v for k, v in names.items()}

    def __call__(self, frame):
        import torch
        t = Timer()
        sync = torch.cuda.synchronize if torch.cuda.is_available() else None
        t.start("pre")
        img, info = self.pre(frame)
        t.stop()
        t.start("infer")
        r = self.model.predict(img, imgsz=self.pre.imgsz, conf=self.conf, half=self.half, verbose=False)[0]
        t.stop(sync)
        t.start("post")
        b = r.boxes
        boxes = Preprocessor.boxes_to_frame(b.xyxy.cpu().numpy(), info)
        res = Result(boxes, b.conf.cpu().numpy().astype(np.float32),
                     np.array([self.class_ids[int(c)] for c in b.cls.cpu().numpy()], np.int32))
        if r.masks is not None and len(r.masks):
            m = r.masks.data.cpu().numpy() > 0.5  # (N, h, w) at model res
            sem = np.zeros(m.shape[1:], np.uint8)
            for i in np.argsort(res.scores):  # high score painted last
                sem[m[i]] = res.classes[i]
            import cv2
            sem = cv2.resize(sem, (self.pre.imgsz, self.pre.imgsz), interpolation=cv2.INTER_NEAREST)
            res.semantic = Preprocessor.semantic_to_frame(sem, info, frame.shape)
        t.stop()
        res.timings = t.t
        return res
