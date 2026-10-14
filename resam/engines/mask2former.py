"""Mask2Former (HuggingFace) backend for the notebook's checkpoints (vit_wradar_*.pth).

Faithful to how those weights were trained:
  * input = left HxH square, inscribed circle, *grayscale replicated to 3 channels*
    (the notebook's evaluation cell fed colour images - a train/test mismatch);
  * 512x512, ImageNet normalisation;
  * the head still has the 80 COCO classes + no-object (num_labels was never changed), so
    predictions with label ids outside the dataset ids (1..6) are discarded here.
"""

import numpy as np

from . import Preprocessor, Result, Timer

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


class Mask2FormerEngine:
    name = "mask2former"

    def __init__(self, weights, imgsz=512, crop="legacy", grayscale=True, conf=0.5, half=True,
                 base_config="facebook/mask2former-swin-base-coco-instance", valid_ids=(1, 2, 3, 4, 5, 6),
                 **_):
        import torch
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

        self.torch = torch
        cfg = Mask2FormerConfig.from_pretrained(base_config)
        model = Mask2FormerForUniversalSegmentation(cfg)
        sd = torch.load(weights, map_location="cpu")
        model.load_state_dict(sd)
        self.dtype = torch.float16 if half else torch.float32
        self.model = model.to("cuda", self.dtype).eval()
        self.processor = Mask2FormerImageProcessor(ignore_index=0, size={"height": imgsz, "width": imgsz})
        self.pre = Preprocessor(imgsz, crop=crop, grayscale=grayscale, rgb=True)
        self.conf = conf
        self.valid_ids = set(valid_ids)
        self.mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
        from ..config import CLASSES
        self.class_names = {k: v for k, v in CLASSES.items() if k in self.valid_ids}

    def __call__(self, frame):
        torch = self.torch
        t = Timer()
        sync = torch.cuda.synchronize
        t.start("pre")
        img, info = self.pre(frame)
        x = torch.from_numpy(img).cuda().permute(2, 0, 1).unsqueeze(0).float()
        x = ((x - self.mean) / self.std).to(self.dtype)
        t.stop(sync)
        t.start("infer")
        with torch.no_grad():
            out = self.model(pixel_values=x)
        t.stop(sync)
        t.start("post")
        out.class_queries_logits = out.class_queries_logits.float()
        out.masks_queries_logits = out.masks_queries_logits.float()
        s = self.pre.imgsz
        pm = self.processor.post_process_instance_segmentation(out, threshold=self.conf, target_sizes=[(s, s)])[0]
        seg = pm["segmentation"]
        sem = torch.zeros((s, s), dtype=torch.uint8, device=seg.device)
        boxes, scores, classes = [], [], []
        for info_seg in pm["segments_info"]:
            lid = int(info_seg["label_id"])
            if lid not in self.valid_ids:
                continue
            m = seg == info_seg["id"]
            if not bool(m.any()):
                continue
            sem[m] = lid
            ys, xs = torch.nonzero(m, as_tuple=True)
            boxes.append([xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1])
            scores.append(float(info_seg["score"]))
            classes.append(lid)
        sem_np = sem.cpu().numpy()
        res = Result(Preprocessor.boxes_to_frame(np.asarray(boxes, np.float32).reshape(-1, 4), info),
                     np.asarray(scores, np.float32), np.asarray(classes, np.int32))
        res.semantic = Preprocessor.semantic_to_frame(sem_np, info, frame.shape)
        if self.pre.crop == "legacy":  # notebook also masked predictions with the circle
            res.semantic *= info["roi"].full_mask(frame.shape)
        t.stop()
        res.timings = t.t
        return res
