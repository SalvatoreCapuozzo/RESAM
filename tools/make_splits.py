#!/usr/bin/env python3
"""Create leakage-aware train/val/test splits (see resam/splits.py) and report how much the
legacy random split leaks (test frames whose temporal neighbour is in train).

  python tools/make_splits.py --strategy blocked --out MasterSplits/splits_blocked.json
  python tools/make_splits.py --strategy session \
      --holdout "IMMAGINI/Cartour delta 1.30/202109150130#82" --out MasterSplits/splits_session.json
  python tools/make_splits.py --check MasterSplits/merged_dataset_splits.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resam.annotations import list_samples  # noqa: E402
from resam.splits import frame_key, make_splits  # noqa: E402

DEFAULT_FOLDERS = [
    "RealImages",
    "IMMAGINI/Cartour delta 00.30/202109150030",
    "IMMAGINI/Cartour delta 1.00/202109150100",
    "IMMAGINI/Cartour delta 1.30/202109150130",
    "IMMAGINI/Lady Laguna 2022 no AIS/202208070330",
]


def leakage(splits, max_dt=31):
    """Fraction of test frames with a same-session train frame within max_dt seconds."""
    train = splits["folds"]["fold_1"]["train"] + splits["folds"]["fold_1"]["val"]
    by_sess = {}
    for b in train:
        s, t = frame_key(b)
        by_sess.setdefault(s, []).append(t)
    hit = 0
    for b in splits["test"]:
        s, t = frame_key(b)
        if any(abs(t - u) <= max_dt for u in by_sess.get(s, [])):
            hit += 1
    return hit, len(splits["test"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folders", nargs="+", default=DEFAULT_FOLDERS)
    ap.add_argument("--strategy", choices=["random", "blocked", "session"], default="blocked")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--block", type=int, default=10, help="frames per contiguous block")
    ap.add_argument("--gap", type=int, default=1, help="frames purged at held-out block edges")
    ap.add_argument("--holdout", nargs="*", default=None, help="session ids for --strategy session")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", default=None, help="only report leakage of an existing split file")
    args = ap.parse_args()

    if args.check:
        with open(args.check) as f:
            sp = json.load(f)
        h, n = leakage(sp)
        print(f"{args.check}: {h}/{n} test frames ({100 * h / max(n, 1):.0f}%) have a train frame "
              f"of the same session within ±31 s (one sweep-sample)")
        return

    bases = list_samples([f for f in args.folders if os.path.isdir(f)])
    sp = make_splits(bases, args.strategy, args.test_frac, args.folds, args.block, args.gap, args.seed,
                     holdout_sessions=args.holdout)
    h, n = leakage(sp)
    print(f"strategy={args.strategy}: unique={sp['meta']['n_unique']} test={n} "
          f"train(f1)={len(sp['folds']['fold_1']['train'])} val(f1)={len(sp['folds']['fold_1']['val'])} "
          f"| temporal leakage {h}/{n} ({100 * h / max(n, 1):.0f}%)")
    out = args.out or f"MasterSplits/splits_{args.strategy}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(sp, f, indent=2)
    print("->", out)


if __name__ == "__main__":
    main()
