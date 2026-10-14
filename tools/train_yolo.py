#!/usr/bin/env python3
"""Train an ultralytics model (YOLO26/YOLO11/YOLOv8/RT-DETR, det or seg) with the RESAM
protocol, so every competitor sees identical data and settings:
  leakage-free split (YOLO_Dataset_v2[_seg]), ROI crop at 960 px, colour input,
  40 epochs, same augmentations, seed 0.

Jetson Orin Nano memory is shared CPU/GPU (7.3 GB): batch 2 and workers 0 are the safe
setting (batch 6 + 2 workers OOM-froze the board). Run under tools/mem_watchdog.sh.

  python tools/train_yolo.py --model yolo11s.pt --name yolo11s_roi960_blocked
  python tools/train_yolo.py --model yolo26s-seg.pt --data YOLO_Dataset_v2_seg/data.yaml --name yolo26s-seg_roi960
  python tools/train_yolo.py --model yolo26s-p2.yaml --init yolo26s.pt --name yolo26s-p2_roi960
"""

import argparse
import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

PROTOCOL = dict(imgsz=960, epochs=40, patience=12, batch=2, workers=0, cache=False, amp=True,
                flipud=0.0, fliplr=0.5, degrees=0.0, scale=0.5, mosaic=1.0, close_mosaic=8,
                seed=0, plots=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="pretrained .pt or model .yaml")
    ap.add_argument("--init", default=None, help="weights to transfer into a .yaml model")
    ap.add_argument("--data", default="/home/jetson/RESAM/YOLO_Dataset_v2/data.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="/home/jetson/RESAM/models/train")
    ap.add_argument("--epochs", type=int, default=PROTOCOL["epochs"])
    ap.add_argument("--batch", type=int, default=PROTOCOL["batch"])
    ap.add_argument("--resume", default=None, help="last.pt of an interrupted run (continues it)")
    args = ap.parse_args()

    if args.resume:
        from ultralytics import RTDETR, YOLO
        (RTDETR if "rtdetr" in args.model else YOLO)(args.resume).train(resume=True)
        return

    if "rtdetr" in os.path.basename(args.model):
        from ultralytics import RTDETR as Model
    else:
        from ultralytics import YOLO as Model
    m = Model(args.model)
    if args.init:
        m = m.load(args.init)
    cfg = dict(PROTOCOL, epochs=args.epochs, batch=args.batch)
    m.train(data=os.path.abspath(args.data), project=args.project, name=args.name, exist_ok=True,
            device=0, **cfg)


if __name__ == "__main__":
    main()
