#!/usr/bin/env python3
"""Evaluate any backend on a split's test set with the full metric suite (resam.metrics).

  python tools/evaluate.py --engine trt:models/yolo26s_fp16_960.engine@crop=roi \
      --splits MasterSplits/splits_blocked.json --mapping visual --out results/eval_yolo26s.json
  python tools/evaluate.py --engine mask2former:vit_wradar_fold_1_20260414_152413.pth \
      --splits MasterSplits/merged_dataset_splits.json --mapping mapping_1
  python tools/evaluate.py --compare results/eval_*.json      # side-by-side table

Ground truth and predictions are compared in full-frame pixel coordinates, inside the
radar circle (annotations outside it are ignored), after mapping both to the chosen
logical class grouping. Outputs a JSON with everything and a Markdown summary.
"""

import argparse
import glob
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resam.annotations import load_sample, map_classes, rasterize_semantic  # noqa: E402
from resam.config import MAPPINGS, profile_for_size, remap_array  # noqa: E402
from resam.engines import create_engine  # noqa: E402
from resam.geometry import roi_for_image  # noqa: E402
from resam.metrics import DetectionEvaluator, LatencyStats, SegEvaluator  # noqa: E402
from resam.splits import frame_key  # noqa: E402
from resam.tracking import SweepTracker  # noqa: E402
from resam.viz import render  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import parse_spec  # noqa: E402


def inside_circle(anns, roi, min_frac=0.5):
    out = []
    for a in anns:
        cx, cy = (a.box[0] + a.box[2]) / 2, (a.box[1] + a.box[3]) / 2
        if math.hypot(cx - roi.cx, cy - roi.cy) <= roi.r * 1.0:
            out.append(a)
    return out


def evaluate(args):
    with open(args.splits) as f:
        splits = json.load(f)
    files = splits[args.subset] if args.subset == "test" else splits["folds"][args.fold][args.subset]
    mp = MAPPINGS[args.mapping]
    names = mp["classes"]
    kind, kw = parse_spec(args.engine)
    # AP needs the whole precision-recall curve: let low-confidence detections through
    # (operating-point metrics still apply --conf). Mask2Former keeps its post-process
    # threshold because its semantic map is painted from the kept segments.
    if kind in ("trt", "ultralytics"):
        kw.setdefault("conf", args.ap_conf)
    eng = create_engine(kind, **kw)

    det = DetectionEvaluator(names)
    det_by_display = defaultdict(lambda: DetectionEvaluator(names))
    seg = SegEvaluator(names)
    lat = LatencyStats()
    roi_cache = {}
    has_sem = None
    tracker, prev_key = None, None
    if args.track_filter:
        # causal persistence filter needs consecutive sweeps: evaluate in time order per session
        files = sorted(files, key=frame_key)
    if args.vis:
        os.makedirs(args.vis, exist_ok=True)
    for i, base in enumerate(files):
        if not os.path.exists(base + ".png"):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = load_sample(base)
        frame = cv2.imread(s.image_path)
        roi = roi_for_image(frame, roi_cache)
        gt = inside_circle(map_classes(s.anns, args.mapping), roi)
        t0 = time.perf_counter()
        r = eng(frame)
        if args.track_filter:
            key = frame_key(base)
            if prev_key is None or key[0] != prev_key[0] or key[1] - prev_key[1] > 31:
                tracker = SweepTracker(confirm_hits=args.track_filter)  # new sequence
            prev_key = key
            r = persistence_filter(tracker, r, args.track_filter)
        lat.add(total=(time.perf_counter() - t0) * 1000, **r.timings)
        pc = np.array([MAPPINGS[args.mapping]["lut"].get(int(c), 0) for c in r.classes], np.int32)
        keep = pc > 0
        pb, ps, pc = r.boxes[keep], r.scores[keep], pc[keep]
        gb = np.array([a.box for a in gt], np.float32).reshape(-1, 4)
        gc = np.array([a.cls for a in gt], np.int32)
        det.add(gb, gc, pb, ps, pc, meta={"name": base})
        disp, _ = profile_for_size(frame.shape[1], frame.shape[0])
        det_by_display[disp or "unknown"].add(gb, gc, pb, ps, pc)
        if r.semantic is not None:
            has_sem = True
            valid = roi.full_mask(frame.shape)
            gsem = rasterize_semantic(gt, frame.shape)
            psem = remap_array(r.semantic, args.mapping)
            seg.add(gsem, psem, valid, name=base)
        if args.vis and i < args.vis_n:
            r2 = r
            r2.boxes, r2.scores, r2.classes = pb, ps, pc
            img = render(frame, r2, names, roi=roi, gt=(gb, gc))
            cv2.imwrite(os.path.join(args.vis, os.path.basename(base) + ".jpg"), img)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)

    out = {"engine": args.engine + (f" +track{args.track_filter}" if args.track_filter else ""), "splits": args.splits, "subset": args.subset, "mapping": args.mapping,
           "detection": det.summary(conf=args.conf),
           "detection_by_display": {k: _display_row(v) for k, v in det_by_display.items()},
           "segmentation": seg.summary() if has_sem else None,
           "latency_ms": lat.summary()}
    return out


def persistence_filter(tracker, r, min_hits):
    """Keep tracked-class detections only once their track has >= min_hits sweeps."""
    tracker.update(r.boxes, r.scores, r.classes)
    keep = np.ones(len(r.boxes), bool)
    if len(r.boxes):
        hits = {tuple(np.round(t.box, 2)): t.hits for t in tracker.active()}
        for i, (b, c) in enumerate(zip(r.boxes, r.classes)):
            if int(c) in tracker.track_classes:
                keep[i] = hits.get(tuple(np.round(b.astype(np.float32), 2)), 0) >= min_hits
    r.boxes, r.scores, r.classes = r.boxes[keep], r.scores[keep], r.classes[keep]
    return r


def _display_row(ev):
    t = ev.ap_table()
    row = {k: t[k] for k in ("mAP50", "mAP50_95", "mAP50_tiny", "mAP50_small")}
    row["n_images"] = len(ev.images)
    return row


def fmt(v):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def markdown(res):
    d = res["detection"]
    L = res["latency_ms"].get("total", {})
    lines = [f"### {res['engine']} — {res['splits']} [{res['subset']}], mapping `{res['mapping']}`", "",
             f"mAP50 **{fmt(d['mAP50'])}** · mAP50-95 {fmt(d['mAP50_95'])} · mAP50 tiny/small/medium/large "
             f"{fmt(d['mAP50_tiny'])}/{fmt(d['mAP50_small'])}/{fmt(d['mAP50_medium'])}/{fmt(d['mAP50_large'])} · "
             f"class-agnostic AP50 {fmt(d['class_agnostic']['AP50'])} · best F1 {fmt(d['best_f1']['F1'])} "
             f"@conf {d['best_f1']['conf']:.2f} · latency p50 {L.get('p50', float('nan')):.1f} ms", "",
             "| class | n_gt | AP50 | AP50-95 | AP50 tiny | AP50 small | P | R (Pd) | FA/frame | Pd centre-hit |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, row in d["per_class"].items():
        op = d["at_conf"]["per_class"][name]
        oc = d["at_conf_center_hit"]["per_class"][name]
        lines.append(f"| {name} | {row['n_gt']} | {fmt(row['AP50'])} | {fmt(row['AP50_95'])} | {fmt(row['AP50_tiny'])} | "
                     f"{fmt(row['AP50_small'])} | {fmt(op['P'])} | {fmt(op['R'])} | {op['FA_per_frame']:.2f} | {fmt(oc['Pd'])} |")
    cm = d["confusion"]
    lines += ["", "Confusion (rows = GT, cols = prediction, IoU ≥ 0.5):", "",
              "| GT \\ pred | " + " | ".join(cm["cols_pred"]) + " |", "|---" * (len(cm["cols_pred"]) + 1) + "|"]
    for rname, row in zip(cm["rows_gt"], cm["matrix"]):
        lines.append(f"| {rname} | " + " | ".join(str(v) for v in row) + " |")
    if res.get("segmentation"):
        s = res["segmentation"]
        lines += ["", f"Segmentation: mIoU **{fmt(s['mIoU'])}** · fwIoU {fmt(s['fwIoU'])} · foreground IoU {fmt(s['fg_IoU'])}",
                  "", "| class | IoU | Dice | Precision | Recall | Boundary F1 |", "|---|---|---|---|---|---|"]
        for name, row in s["per_class"].items():
            if row:
                lines.append(f"| {name} | {fmt(row['IoU'])} | {fmt(row['Dice'])} | {fmt(row['Precision'])} | "
                             f"{fmt(row['Recall'])} | {fmt(row['BoundaryF1'])} |")
        o = s["object_level_center_hit"]["overall"]
        lines.append(f"\nObject level (connected components, centre-hit): Pd {fmt(o['Pd'])}, "
                     f"precision {fmt(o['P'])}, FA/frame {o['FA_per_frame']:.2f}")
    return "\n".join(lines)


def compare(paths):
    rows = ["| engine | split | mapping | mAP50 | mAP50-95 | AP50 tiny | AP50 small | agnostic AP50 | best F1 | mIoU | p50 ms |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in paths:
        r = json.load(open(p))
        d = r["detection"]
        s = r.get("segmentation") or {}
        rows.append(f"| {r['engine']} | {os.path.basename(r['splits'])} | {r['mapping']} | {fmt(d['mAP50'])} | "
                    f"{fmt(d['mAP50_95'])} | {fmt(d['mAP50_tiny'])} | {fmt(d['mAP50_small'])} | "
                    f"{fmt(d['class_agnostic']['AP50'])} | {fmt(d['best_f1']['F1'])} | {fmt(s.get('mIoU'))} | "
                    f"{r['latency_ms'].get('total', {}).get('p50', float('nan')):.1f} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine")
    ap.add_argument("--splits", default="MasterSplits/splits_blocked.json")
    ap.add_argument("--subset", default="test", choices=["test", "val", "train"])
    ap.add_argument("--fold", default="fold_1")
    ap.add_argument("--mapping", default="original", choices=list(MAPPINGS))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--ap-conf", type=float, default=0.001, help="engine score floor used for AP curves")
    ap.add_argument("--track-filter", type=int, default=0,
                    help="keep target detections only after N consecutive-sweep track hits (0 = off)")
    ap.add_argument("--vis", default=None, help="folder for overlay images (GT white, preds coloured)")
    ap.add_argument("--vis-n", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs="+", default=None)
    args = ap.parse_args()

    if args.compare:
        paths = sorted({p for pat in args.compare for p in glob.glob(pat)})
        print(compare(paths))
        return
    res = evaluate(args)
    md = markdown(res)
    print(md)
    out = args.out or f"results/eval_{args.engine.split(':')[0]}_{args.mapping}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    with open(os.path.splitext(out)[0] + ".md", "w") as f:
        f.write(md + "\n")
    print("->", out)


if __name__ == "__main__":
    main()
