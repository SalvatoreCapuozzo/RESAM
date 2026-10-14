"""Leakage-aware dataset splits for sequential radar frames.

Frames are consecutive sweeps (~15 s apart) named R_<MMSS><session-suffix>. A random
per-frame split puts near-identical neighbours in train and test and inflates every
metric. Strategies here:
  * random   - legacy behaviour (per-frame), kept for comparison only
  * blocked  - contiguous time blocks per session, assigned to splits, with a purge gap
  * session  - hold out whole recording sessions (strictest: tests generalisation)
Identical image files (e.g. RealImages/*53 == 'Lady Laguna' frames) are de-duplicated.
"""

import hashlib
import os
import random
import re
from collections import defaultdict

_NAME_RE = re.compile(r"R_(\d{2})(\d{2})(\d{2})$")


def frame_key(base_path):
    """-> (session_id, seconds_within_hour) parsed from folder + filename."""
    name = os.path.basename(base_path)
    m = _NAME_RE.match(name)
    folder = os.path.dirname(base_path)
    if not m:
        return folder, 0
    mm, ss, suffix = m.groups()
    return f"{folder}#{suffix}", int(mm) * 60 + int(ss)


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dedupe(base_paths, image_ext=".png"):
    seen, keep, dups = {}, [], []
    for b in base_paths:
        h = file_md5(b + image_ext)
        if h in seen:
            dups.append((b, seen[h]))
        else:
            seen[h] = b
            keep.append(b)
    return keep, dups


def group_sessions(base_paths):
    sessions = defaultdict(list)
    for b in base_paths:
        sid, t = frame_key(b)
        sessions[sid].append((t, b))
    return {k: [b for _, b in sorted(v)] for k, v in sessions.items()}


def _blocks(sessions, block):
    out = []
    for sid, frames in sorted(sessions.items()):
        for i in range(0, len(frames), block):
            out.append((sid, frames[i:i + block]))
    return out


def make_splits(base_paths, strategy="blocked", test_frac=0.2, n_folds=5, block=10, gap=1,
                seed=42, holdout_sessions=None, image_ext=".png"):
    """Return {'test': [...], 'folds': {'fold_k': {'train': [...], 'val': [...]}}, 'meta': {...}}
    in the same layout as MasterSplits/merged_dataset_splits.json."""
    base_paths, dups = dedupe(base_paths, image_ext)
    rng = random.Random(seed)
    meta = {"strategy": strategy, "block": block, "gap": gap, "seed": seed,
            "n_unique": len(base_paths), "duplicates_removed": [d[0] for d in dups]}

    if strategy == "random":
        items = list(base_paths)
        rng.shuffle(items)
        n_test = int(round(test_frac * len(items)))
        test, trainval = items[:n_test], items[n_test:]
        units = [[b] for b in trainval]
    else:
        sessions = group_sessions(base_paths)
        meta["sessions"] = {k: len(v) for k, v in sessions.items()}
        if strategy == "session":
            hold = set(holdout_sessions or [])
            if not hold:
                raise ValueError("strategy='session' needs holdout_sessions")
            test = [b for s in sorted(hold) for b in sessions[s]]
            rest = {k: v for k, v in sessions.items() if k not in hold}
            units = [frames for _, frames in _blocks(rest, block)]
        elif strategy == "blocked":
            blocks = _blocks(sessions, block)
            rng.shuffle(blocks)
            # stratify by session: take ~test_frac of every session's blocks
            per_sess = defaultdict(list)
            for sid, frames in blocks:
                per_sess[sid].append(frames)
            test, units = [], []
            for sid, bl in sorted(per_sess.items()):
                n_test = max(1, int(round(test_frac * len(bl)))) if len(bl) > 1 else 0
                for frames in bl[:n_test]:
                    test.extend(_trim(frames, gap))
                units.extend(bl[n_test:])
        else:
            raise ValueError(f"unknown strategy {strategy}")

    rng.shuffle(units)
    folds = {}
    for k in range(n_folds):
        val_units = units[k::n_folds]
        tr_units = [u for i, u in enumerate(units) if i % n_folds != k]
        folds[f"fold_{k + 1}"] = {
            "train": [b for u in tr_units for b in u],
            "val": [b for u in val_units for b in _trim(u, gap if strategy != "random" else 0)],
        }
    return {"test": test, "folds": folds, "meta": meta}


def _trim(frames, gap):
    """Remove `gap` frames from each end of a block that is held out (purged CV)."""
    if gap <= 0 or len(frames) <= 2 * gap:
        return list(frames)
    return list(frames[gap:len(frames) - gap])
