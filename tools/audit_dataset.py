#!/usr/bin/env python3
"""Dataset audit: everything that silently degrades training or inflates metrics.

  python tools/audit_dataset.py --out results/dataset_audit.json

Reports per folder/session: images, displays (resolution), class counts and box sizes;
invalid class ids; basename collisions across folders; byte-identical duplicate images;
temporal leakage of a split file; and, per class, how many boxes contain radar-echo
pixels vs. display vector glyphs (AIS triangles) - i.e. whether a class is visually
learnable from the screen at all.
"""

import argparse
import collections
import json
import os
import sys
import warnings

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_splits import DEFAULT_FOLDERS, leakage  # noqa: E402
from resam.annotations import list_samples, load_sample  # noqa: E402
from resam.config import CLASSES  # noqa: E402
from resam.echo import echo_mask  # noqa: E402
from resam.splits import dedupe, frame_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folders", nargs="+", default=DEFAULT_FOLDERS)
    ap.add_argument("--splits", nargs="*", default=["MasterSplits/merged_dataset_splits.json"])
    ap.add_argument("--no-pixels", action="store_true", help="skip the echo/glyph content analysis")
    ap.add_argument("--out", default="results/dataset_audit.json")
    args = ap.parse_args()

    folders = [f for f in args.folders if os.path.isdir(f)]
    bases = list_samples(folders)
    rep = {"n_annotated": len(bases), "folders": {}, "invalid": [], "content": {}}
    by_name = collections.defaultdict(list)
    content = collections.defaultdict(collections.Counter)
    sizes = collections.defaultdict(list)
    for b in bases:
        by_name[os.path.basename(b)].append(os.path.dirname(b))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = load_sample(b)
        if s.dropped:
            rep["invalid"].append({"file": b, "dropped": s.dropped})
        f = rep["folders"].setdefault(os.path.dirname(b), {"images": 0, "resolutions": collections.Counter(),
                                                           "classes": collections.Counter(), "types": collections.Counter()})
        f["images"] += 1
        f["resolutions"][f"{s.width}x{s.height}"] += 1
        for a in s.anns:
            f["classes"][CLASSES[a.cls]] += 1
            f["types"][a.kind] += 1
            sizes[CLASSES[a.cls]].append(float(np.sqrt(max(a.area, 1))))
        if not args.no_pixels:
            im = cv2.imread(s.image_path)
            em = echo_mask(im) > 0
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            glyph = (hsv[..., 1] < 60) & (hsv[..., 2] > 120)
            for a in s.anns:
                x0, y0, x1, y1 = np.round(a.box).astype(int)
                x0, y0 = max(x0, 0), max(y0, 0)
                if x1 <= x0 or y1 <= y0:
                    continue
                e, g = em[y0:y1, x0:x1].mean(), glyph[y0:y1, x0:x1].mean()
                k = ("echo+glyph" if g > 0.02 else "echo only") if e > 0.03 else ("glyph only" if g > 0.02 else "neither")
                content[CLASSES[a.cls]][k] += 1

    rep["collisions"] = {k: v for k, v in by_name.items() if len(v) > 1}
    _, dups = dedupe(bases)
    rep["identical_duplicates"] = [{"file": a, "same_as": b} for a, b in dups]
    rep["sessions"] = dict(collections.Counter(frame_key(b)[0] for b in bases))
    rep["size_sqrt_area_px"] = {k: {"p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
                                    "p90": float(np.percentile(v, 90)), "n": len(v)} for k, v in sizes.items()}
    rep["content"] = {k: {kk: round(100 * vv / sum(v.values()), 1) for kk, vv in v.items()} for k, v in content.items()}
    rep["split_leakage"] = {}
    for sp in args.splits:
        if os.path.exists(sp):
            h, n = leakage(json.load(open(sp)))
            rep["split_leakage"][sp] = {"test_frames_with_train_neighbour_31s": h, "test_frames": n}

    print(f"annotated images: {len(bases)} | sessions: {len(rep['sessions'])}")
    print(f"invalid labels: {len(rep['invalid'])} files | basename collisions: {len(rep['collisions'])} | "
          f"identical duplicate images: {len(dups)}")
    for k, v in rep["split_leakage"].items():
        print(f"leakage {k}: {v['test_frames_with_train_neighbour_31s']}/{v['test_frames']} test frames")
    print("object size sqrt(area) px (p10 / median / p90):")
    for k, v in rep["size_sqrt_area_px"].items():
        print(f"  {k:12s} {v['p10']:5.0f} {v['median']:5.0f} {v['p90']:5.0f}  (n={v['n']})")
    if rep["content"]:
        print("box content (echo pixels vs AIS/vector glyph pixels):")
        for k, v in rep["content"].items():
            print(f"  {k:12s} " + "  ".join(f"{kk}: {vv:.0f}%" for kk, vv in sorted(v.items())))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2, default=lambda o: dict(o) if isinstance(o, collections.Counter) else str(o))
    print("->", args.out)


if __name__ == "__main__":
    main()
