#!/usr/bin/env python3
"""Build a YOLO detection or segmentation dataset from the hybrid bbox/polygon labels.

Differences from the notebook's build_yolo_bbox_dataset():
  * unique, folder-prefixed filenames (300 basenames collide across folders; the old
    builder silently overwrote images/labels from different sessions)
  * invalid class ids are dropped (R_283282 had class 53 -> YOLO class 52 -> corrupt label)
  * --crop roi: tight crop around the measured radar ring (default); --crop legacy keeps
    the left-square + inscribed-circle behaviour for comparison with old weights
  * --task segment: polygons kept; bbox-only echo labels become echo-shaped pseudo-masks
  * --mapping: train directly on a logical class grouping (e.g. 'visual')
  * images keep their colour (the radar palette separates echoes from overlays)

Example:
  python tools/build_yolo_dataset.py --splits MasterSplits/splits_blocked.json \
      --out YOLO_Dataset_det --task detect --mapping original
"""

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resam.annotations import (load_sample, map_classes, refine_boxes_with_echo, to_roi,  # noqa: E402
                               unique_name, yolo_det_lines, yolo_seg_lines)
from resam.config import MAPPINGS  # noqa: E402
from resam.geometry import RadarROI, roi_for_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="MasterSplits/merged_dataset_splits.json")
    ap.add_argument("--fold", default="fold_1", help="fold whose train/val define the YOLO train/val")
    ap.add_argument("--trainval-all", action="store_true", help="use train+val of the fold as train (final model)")
    ap.add_argument("--out", default="YOLO_Dataset_v2")
    ap.add_argument("--task", choices=["detect", "segment"], default="detect")
    ap.add_argument("--crop", choices=["roi", "legacy"], default="roi")
    ap.add_argument("--mapping", default="original", choices=list(MAPPINGS))
    ap.add_argument("--refine-boxes", action="store_true", help="(segment) bbox -> echo pseudo-mask")
    ap.add_argument("--ext", default=".png")
    args = ap.parse_args()

    with open(args.splits) as f:
        splits = json.load(f)
    fold = splits["folds"][args.fold]
    sets = {"train": fold["train"] + (fold["val"] if args.trainval_all else []),
            "val": fold["val"], "test": splits["test"]}
    classes = MAPPINGS[args.mapping]["classes"]
    # contiguous 0-based YOLO ids for the chosen mapping
    yolo_ids = {cid: i for i, cid in enumerate(sorted(classes))}
    report = {"dropped": [], "counts": {}}
    roi_cache = {}

    for split, bases in sets.items():
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)
        n_obj = 0
        for base in bases:
            if not os.path.exists(base + args.ext):
                report["dropped"].append({"file": base, "reason": "missing image"})
                continue
            s = load_sample(base, args.ext)
            if s.dropped:
                report["dropped"].append({"file": base, "annotations": s.dropped})
            img = cv2.imread(s.image_path)
            h = img.shape[0]
            roi = roi_for_image(img, roi_cache) if args.crop == "roi" else RadarROI(h / 2, h / 2, h / 2, margin=1.0)
            anns = s.anns
            if args.task == "segment" and args.refine_boxes:
                anns = refine_boxes_with_echo(img, anns)
            anns = to_roi(map_classes(anns, args.mapping), roi)
            crop = roi.crop(img)
            side = crop.shape[0]
            name = unique_name(base)
            cv2.imwrite(os.path.join(args.out, "images", split, name + ".png"), crop)
            for a in anns:
                a.cls = yolo_ids[a.cls] + 1  # exporters subtract 1
            lines = (yolo_seg_lines if args.task == "segment" else yolo_det_lines)(anns, side, side)
            with open(os.path.join(args.out, "labels", split, name + ".txt"), "w") as f:
                f.write("\n".join(lines))
            n_obj += len(lines)
        report["counts"][split] = {"images": len(bases), "objects": n_obj}

    names = "\n".join(f"  {yolo_ids[c]}: {classes[c]}" for c in sorted(classes))
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                f"names:\n{names}\n")
    meta = {"task": args.task, "crop": args.crop, "mapping": args.mapping, "splits": args.splits,
            "fold": args.fold, "yolo_to_dataset_id": {v: k for k, v in yolo_ids.items()}, **report}
    with open(os.path.join(args.out, "build_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(report["counts"]), f"dropped entries: {len(report['dropped'])}")
    print(f"-> {args.out}/data.yaml")


if __name__ == "__main__":
    main()
