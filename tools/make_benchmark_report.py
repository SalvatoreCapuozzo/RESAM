#!/usr/bin/env python3
"""Build docs/BENCHMARK.md: competitor leaderboard (mAP50 / mask mAP50 + FPS on the Jetson),
charts and inference grids for several radar use cases.

Inputs (produced by tools/run_competitors.py and the earlier runs):
  results/competitors/<name>_{ultra_val,bench,eval_original,eval_visual}.json
  results/eval_y26s_int8mix_blocked_*.json, results/benchmark_final_s.json   (YOLO26s INT8-mixed)
  results/eval_classical_blocked_*.json, results/benchmark_orin_nano.json     (classical, Mask2Former)
  results/eval_m2f_legacy_original_colour.json                                 (Mask2Former, legacy split)

  python tools/make_benchmark_report.py            # tables + charts + use-case images
  python tools/make_benchmark_report.py --no-images
"""

import argparse
import json
import math
import os
import sys
import warnings

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from resam.annotations import load_sample  # noqa: E402
from resam.config import CLASSES  # noqa: E402
from resam.engines import create_engine  # noqa: E402
from resam.geometry import roi_for_image  # noqa: E402
from resam.viz import CLASS_COLORS  # noqa: E402

RES = os.path.join(ROOT, "results")
COMP = os.path.join(RES, "competitors")
IMG = os.path.join(ROOT, "docs", "img", "bench")
OUT_MD = os.path.join(ROOT, "docs", "BENCHMARK.md")

# dataviz reference palette, light mode; only the first 3 categorical slots (validated all-pairs
# for scatter use), every point is also direct-labelled and marker shape encodes the task.
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8d8c84", "#e4e3df", "#fcfcfb"
FAMILY_COLOR = {"YOLO26": "#2a78d6", "YOLO11 / v8": "#eb6834", "Transformer / classical": "#1baf7a"}
SEQ_CMAP = "Blues"

USE_CASES = [
    ("clutter", "Dense sea clutter", "RealImages/R_335353",
     "Green 1024×768 console at 1.5 NM: ~25 labelled clutter specks of 1–3 px and a few real echoes."),
    ("coast", "Coastal / port approach", "IMMAGINI/Cartour delta 00.30/202109150030/R_573282",
     "Orange console at 0.5 NM, off-centre: large coastline returns with targets close to the shore."),
    ("traffic", "Multi-target traffic, small echoes", "RealImages/R_261570",
     "15 labelled targets, 10 of them smaller than 20 px — the tiny-object regime."),
    ("ais", "AIS-tracked traffic", "RealImages/R_243070",
     "AIS triangles drawn by the console next to (or without) radar echoes."),
    ("open_sea", "Open sea, sparse long-range targets", "IMMAGINI/Cartour delta 1.30/202109150130/R_423282",
     "6 NM range, own ship under way: few isolated echoes, heading/EBL lines and AIS symbols on screen."),
]


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def models():
    """Registry of every competitor with whatever results exist."""
    m = []
    status = jload(os.path.join(COMP, "status.json")) or {}
    meta = {
        "yolo26n": ("YOLO26n", "YOLO26", "det"), "yolo26s": ("YOLO26s", "YOLO26", "det"),
        "yolo26s-p2": ("YOLO26s-P2", "YOLO26", "det"), "yolo11s": ("YOLO11s", "YOLO11 / v8", "det"),
        "yolov8s": ("YOLOv8s", "YOLO11 / v8", "det"), "rtdetr-l": ("RT-DETR-L", "Transformer / classical", "det"),
        "yolo26s-seg": ("YOLO26s-seg", "YOLO26", "seg"), "yolo11s-seg": ("YOLO11s-seg", "YOLO11 / v8", "seg"),
    }
    for key, (label, fam, task) in meta.items():
        r = {"key": key, "label": label + " FP16", "family": fam, "task": task,
             "val": jload(os.path.join(COMP, f"{key}_ultra_val.json")),
             "bench": jload(os.path.join(COMP, f"{key}_bench.json")),
             "eval": jload(os.path.join(COMP, f"{key}_eval_original.json")),
             "eval_visual": jload(os.path.join(COMP, f"{key}_eval_visual.json")),
             "engine": f"trt:models/engines/{key}_roi960_fp16.engine@crop=roi",
             "status": status.get(key, {})}
        if r["eval"] or r["val"]:
            m.append(r)
    # YOLO26s INT8-mixed (deployment config)
    fs = jload(os.path.join(RES, "benchmark_final_s.json"))
    b8 = next((x for x in (fs or {}).get("results", []) if "int8mix" in x["engine"]), None)
    m.append({"key": "yolo26s-int8mix", "label": "YOLO26s INT8-mixed", "family": "YOLO26", "task": "det",
              "val": None, "bench": {"results": [b8]} if b8 else None,
              "eval": jload(os.path.join(RES, "eval_y26s_int8mix_blocked_original.json")),
              "eval_visual": jload(os.path.join(RES, "eval_y26s_int8mix_blocked_visual.json")),
              "engine": "trt:models/engines/yolo26s_roi960_int8mix.engine@crop=roi"})
    on = jload(os.path.join(RES, "benchmark_orin_nano.json")) or {"results": []}
    cl = next((x for x in on["results"] if x["engine"] == "classical"), None)
    m.append({"key": "classical", "label": "Classical echo blobs", "family": "Transformer / classical", "task": "det",
              "val": None, "bench": {"results": [cl]} if cl else None,
              "eval": jload(os.path.join(RES, "eval_classical_blocked_original.json")),
              "eval_visual": jload(os.path.join(RES, "eval_classical_blocked_visual.json")), "engine": "classical"})
    mf = next((x for x in on["results"] if x["engine"].startswith("mask2former")), None)
    m.append({"key": "mask2former", "label": "Mask2Former Swin-B † (box mAP50)", "family": "Transformer / classical",
              "task": "seg", "val": None, "bench": {"results": [mf]} if mf else None,
              "eval": jload(os.path.join(RES, "eval_m2f_legacy_original_colour.json")), "eval_visual": None,
              "engine": "mask2former:vit_wradar_fold_1_20260414_152413.pth@grayscale=0", "legacy": True})
    for r in m:
        b = (r["bench"] or {}).get("results", [None])[0] if r["bench"] else None
        r["b"] = b if b and "error" not in b else None
    return m


def g(d, *path, default=float("nan")):
    for k in path:
        if d is None or k not in d:
            return default
        d = d[k]
    return d if d is not None else default


def f3(v):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def f1(v):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.1f}"


# ------------------------------------------------------------------------------ charts
def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_pareto(ms, path):
    pts = [(r["label"], g(r["b"], "throughput_fps"), metric_primary(r), r["family"], r["task"]) for r in ms
           if r["b"] and not math.isnan(metric_primary(r))]
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    style(ax, "Accuracy vs. speed on Jetson Orin Nano (15 W)", "Throughput, frames/s (log scale)",
          "mAP50 (detection) / mask mAP50 (segmentation)")
    ax.set_xscale("log")
    ax.axvspan(0.3, 0.8, color=GRID, alpha=0.6, lw=0)
    ax.text(0.55, 0.02, "slower than\none sweep\n(1.25 s)", ha="center", va="bottom", fontsize=8, color=MUTED,
            transform=ax.get_xaxis_transform())
    for lab, fps, acc, fam, task in pts:
        ax.scatter(fps, acc, s=70, color=FAMILY_COLOR[fam], marker="o" if task == "det" else "s",
                   edgecolor=SURFACE, linewidth=2, zorder=3)
    # Pareto front (maximise both)
    front, best = [], -1
    for lab, fps, acc, fam, task in sorted(pts, key=lambda p: -p[1]):
        if acc > best:
            front.append((fps, acc))
            best = acc
    if len(front) > 1:
        xs, ys = zip(*sorted(front))
        ax.plot(xs, ys, ls="--", lw=1.2, color=MUTED, zorder=2)
    ax.set_xlim(0.3, max(p[1] for p in pts) * 2.2)
    ax.set_ylim(0, 1)
    place_labels(fig, ax, [(p[0], p[1], p[2]) for p in pts])
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAMILY_COLOR.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", color=MUTED, label="detection (box mAP50)"),
                plt.Line2D([], [], marker="s", ls="", color=MUTED, label="segmentation (mask mAP50)"),
                plt.Line2D([], [], ls="--", color=MUTED, label="Pareto front")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def place_labels(fig, ax, items, fontsize=8):
    """Direct labels without collisions: try offsets around each point, keep the first whose
    text box overlaps neither placed labels nor other markers; fall back to a leader line."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts_px = [ax.transData.transform((x, y)) for _, x, y in items]
    placed = []
    offsets = [(7, 4), (7, -11), (-7, 4), (-7, -11), (7, 14), (7, -21), (-7, 14), (-7, -21),
               (14, 24), (14, -31), (-14, 24), (-14, -31), (20, 36), (20, -43)]

    def overlaps(bb):
        for o in placed:
            if bb.x0 < o.x1 and bb.x1 > o.x0 and bb.y0 < o.y1 and bb.y1 > o.y0:
                return True
        for px, py in pts_px:
            if bb.x0 - 4 < px < bb.x1 + 4 and bb.y0 - 4 < py < bb.y1 + 4:
                return True
        return False

    for (lab, x, y), _ in sorted(zip(items, pts_px), key=lambda t: -t[0][2]):
        for dx, dy in offsets:
            t = ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                            color=INK, ha="left" if dx > 0 else "right",
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6) if abs(dy) > 14 else None)
            bb = t.get_window_extent(rend)
            if not overlaps(bb):
                placed.append(bb)
                break
            t.remove()
        else:
            placed.append(ax.annotate(lab, (x, y), xytext=(7, 4), textcoords="offset points",
                                      fontsize=fontsize, color=INK).get_window_extent(rend))


def chart_heatmap(rows, cols, data, title, path, fmt="{:.2f}"):
    data = np.array(data, float)
    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 3, 0.42 * len(rows) + 1.4), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    masked = np.ma.masked_invalid(data)
    ax.imshow(masked, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, color=INK2, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, color=INK, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = data[i, j]
            txt = "–" if math.isnan(v) else fmt.format(v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if (not math.isnan(v) and v > 0.55) else INK)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def chart_latency(ms, path):
    rows = [r for r in ms if r["b"]]
    rows.sort(key=lambda r: g(r["b"], "latency_ms", "total", "p50"))
    stages = [("pre", "pre-process (crop, resize, upload)", "#2a78d6"),
              ("infer", "inference", "#eb6834"), ("post", "decode / post-process", "#1baf7a")]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.6), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    style(ax, "Latency per frame, p50 (ms)", "milliseconds (log scale)", "")
    ax.set_xscale("log")
    y = np.arange(len(rows))
    left = np.full(len(rows), 0.5)
    for key, lab, col in stages:
        v = np.array([max(g(r["b"], "latency_ms", key, "p50", default=0.0), 0.01) for r in rows])
        ax.barh(y, v, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.62, label=lab)
        left = left + v
    for i, r in enumerate(rows):
        ax.text(left[i] * 1.05, i, f"{g(r['b'], 'latency_ms', 'total', 'p50'):.0f} ms", va="center",
                fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], color=INK, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.5, left.max() * 3)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def chart_energy(ms, path):
    rows = [r for r in ms if r["b"] and r["b"].get("energy_mj_per_frame")]
    rows.sort(key=lambda r: r["b"]["energy_mj_per_frame"])
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.4), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    style(ax, "Energy per processed frame (board input, mJ)", "millijoules (log scale)", "")
    ax.set_xscale("log")
    v = [r["b"]["energy_mj_per_frame"] for r in rows]
    ax.barh(range(len(rows)), v, color="#2a78d6", height=0.62, edgecolor=SURFACE, linewidth=2)
    for i, x in enumerate(v):
        ax.text(x * 1.06, i, f"{x:,.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], color=INK, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(v) / 2, max(v) * 3)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def metric_primary(r):
    """Headline accuracy: detection -> box mAP50 (RESAM evaluator, same code for all models);
    segmentation -> instance mask mAP50 (ultralytics val), Mask2Former -> its box mAP50 (no masks AP)."""
    if r["task"] == "seg" and r["val"] and "mask_map50" in r["val"]:
        return r["val"]["mask_map50"]
    return g(r["eval"], "detection", "mAP50")


# ------------------------------------------------------------------------------ images
def tile(frame, roi, res, label, sub, gt=None, side=500):
    """ROI crop resized first, then boxes drawn at display scale (readable at any zoom).
    Predictions: class-coloured 2 px boxes (+ masks for segmentation models); GT: thin white."""
    crop = roi.crop(frame)
    k = side / crop.shape[0]
    img = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
    x0, y0 = roi.crop_box[:2]
    if res is not None and res.semantic is not None:
        x1, y1 = x0 + crop.shape[1], y0 + crop.shape[0]
        sem = np.zeros(crop.shape[:2], np.uint8)
        h, w = res.semantic.shape
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        sem[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = res.semantic[sy0:sy1, sx0:sx1]
        sem = cv2.resize(sem, (side, side), interpolation=cv2.INTER_NEAREST)
        col = np.zeros_like(img)
        for c in np.unique(sem):
            if c:
                col[sem == c] = CLASS_COLORS.get(int(c), (255, 255, 255))
        m = sem > 0
        img[m] = (img[m] * 0.35 + col[m] * 0.65).astype(np.uint8)

    def draw(boxes, classes, gt_style):
        for bb, c in zip(boxes, classes):
            p0 = (int((bb[0] - x0) * k), int((bb[1] - y0) * k))
            p1 = (int((bb[2] - x0) * k), int((bb[3] - y0) * k))
            if gt_style:
                cv2.rectangle(img, p0, p1, (235, 235, 235), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(img, p0, p1, CLASS_COLORS.get(int(c), (255, 255, 255)), 2, cv2.LINE_AA)

    if gt is not None:
        if res is None:  # ground-truth tile: class colours
            draw(*gt, gt_style=False)
        else:
            draw(*gt, gt_style=True)
    if res is not None:
        draw(res.boxes, res.classes, gt_style=False)
    bar = np.full((48, side, 3), 250, np.uint8)
    cv2.putText(bar, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (11, 11, 11), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (78, 81, 82), 1, cv2.LINE_AA)
    return np.vstack([bar, img, np.full((6, side, 3), 250, np.uint8)])


def legend_strip(width):
    bar = np.full((30, width, 3), 250, np.uint8)
    x = 10
    for cid, name in list(CLASSES.items()) + [(0, "thin white = ground truth (model tiles)")]:
        col = CLASS_COLORS.get(cid, (255, 255, 255)) if cid else (60, 60, 60)
        cv2.rectangle(bar, (x, 9), (x + 14, 21), col, -1)
        cv2.putText(bar, name, (x + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
        x += 30 + int(cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0])
    return bar


def use_case_images(ms, conf=0.25, only=None):
    from benchmark import parse_spec
    frames = {}
    for key, title, base, _ in USE_CASES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = load_sample(os.path.join(ROOT, base))
        fr = cv2.imread(s.image_path)
        frames[key] = (fr, s, roi_for_image(fr, {}))
    tiles = {k: [] for k in frames}
    for k, (fr, s, roi) in frames.items():
        gb = np.array([a.box for a in s.anns], np.float32).reshape(-1, 4)
        gc = np.array([a.cls for a in s.anns], np.int32)
        tiles[k].append(tile(fr, roi, None, "Ground truth", f"{len(gc)} labelled objects", gt=(gb, gc)))
    per_model = {}
    for r in ms:
        if only and r["key"] not in only:
            continue
        kind, kw = parse_spec(r["engine"])
        if kind in ("trt", "ultralytics"):
            kw["conf"] = conf
        try:
            eng = create_engine(kind, **kw)
        except Exception as e:  # engine missing (job not finished)
            print("skip", r["label"], e)
            continue
        for k, (fr, s, roi) in frames.items():
            eng(fr)  # warm-up
            res = eng(fr)
            gb = np.array([a.box for a in s.anns], np.float32).reshape(-1, 4)
            gc = np.array([a.cls for a in s.anns], np.int32)
            t = sum(res.timings.get(x, 0) for x in ("pre", "infer", "post"))
            per_model.setdefault(r["key"], {})[k] = len(res.boxes)
            tiles[k].append(tile(fr, roi, res, r["label"], f"{len(res.boxes)} detections  |  {t:.0f} ms",
                                 gt=(gb, gc)))
        del eng
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
    out = {}
    for k, ts in tiles.items():
        cols = 4
        while len(ts) % cols:
            ts.append(np.full_like(ts[0], 250))
        grid = np.vstack([np.hstack(ts[i:i + cols]) for i in range(0, len(ts), cols)])
        grid = np.vstack([grid, legend_strip(grid.shape[1])])
        p = os.path.join(IMG, f"uc_{k}.jpg")
        cv2.imwrite(p, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out[k] = p
    return out, per_model


# ------------------------------------------------------------------------------ markdown
def status_section():
    import datetime
    st = jload(os.path.join(COMP, "status.json")) or {}
    order = ["yolo26s", "yolo26n", "yolo11s", "yolov8s", "yolo26s-seg", "yolo11s-seg", "yolo26s-p2", "rtdetr-l"]
    log = os.path.join(ROOT, "models", "competitors.log")
    all_done = os.path.exists(log) and "ALL_DONE" in open(log).read()
    rows = ["| Competitor | Status |", "|---|---|"]
    for k in order:
        s = st.get(k, {})
        if os.path.exists(os.path.join(COMP, f"{k}_eval_visual.json")):
            state = "✅ done"
        elif any(v not in ("ok",) for v in s.values()):
            state = "⚠️ failed: " + ", ".join(f"{a}={b}" for a, b in s.items() if b != "ok")
        elif s or k == next((o for o in order if not os.path.exists(os.path.join(COMP, f"{o}_eval_visual.json"))), None):
            state = "⏳ running"
        else:
            state = "queued"
        rows.append(f"| {k} | {state} |")
    head = ("**Status: complete.**" if all_done else
            "**Status: in progress** — this file is regenerated automatically each time a competitor finishes; "
            "inference images are refreshed at the end.")
    return "\n".join([f"> {head} Last update {datetime.datetime.now():%Y-%m-%d %H:%M}.", "", *rows])


def det_table(ms):
    rows = ["| Model | Params (M) | mAP50 ↑ | mAP50 (ultralytics val) | mAP50-95 | AP50 tiny <16 px | AP50 small 16–32 px "
            "| Pd centre-hit | False alarms / frame | FPS ↑ | p50 / p95 ms | Power W | mJ / frame |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted([r for r in ms if r["task"] == "det"], key=lambda r: -metric_primary(r) if not math.isnan(metric_primary(r)) else 1):
        d = g(r["eval"], "detection", default=None)
        b = r["b"]
        rows.append(
            f"| {r['label']} | {f1(g(r['val'], 'params_M'))} | **{f3(g(d, 'mAP50'))}** | {f3(g(r['val'], 'box_map50'))} | "
            f"{f3(g(d, 'mAP50_95'))} | {f3(g(d, 'mAP50_tiny'))} | {f3(g(d, 'mAP50_small'))} | "
            f"{f3(g(d, 'at_conf_center_hit', 'overall', 'Pd'))} | {f1(g(d, 'at_conf', 'overall', 'FA_per_frame'))} | "
            f"**{f1(g(b, 'throughput_fps'))}** | {f1(g(b, 'latency_ms', 'total', 'p50'))} / "
            f"{f1(g(b, 'latency_ms', 'total', 'p95'))} | {f1(g(b, 'power_mw_mean') / 1000 if b and b.get('power_mw_mean') else float('nan'))} | "
            f"{f1(g(b, 'energy_mj_per_frame'))} |")
    return "\n".join(rows)


def seg_table(ms):
    rows = ["| Model | Params (M) | **mask mAP50** ↑ | mask mAP50-95 | box mAP50 | mIoU (semantic) | foreground IoU "
            "| FPS ↑ | p50 / p95 ms | Power W | mJ / frame |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in [r for r in ms if r["task"] == "seg"]:
        b, s = r["b"], g(r["eval"], "segmentation", default=None)
        rows.append(
            f"| {r['label']} | {f1(g(r['val'], 'params_M'))} | **{f3(g(r['val'], 'mask_map50'))}** | "
            f"{f3(g(r['val'], 'mask_map50_95'))} | {f3(g(r['eval'], 'detection', 'mAP50'))} | {f3(g(s, 'mIoU'))} | "
            f"{f3(g(s, 'fg_IoU'))} | **{f1(g(b, 'throughput_fps'))}** | {f1(g(b, 'latency_ms', 'total', 'p50'))} / "
            f"{f1(g(b, 'latency_ms', 'total', 'p95'))} | "
            f"{f1(g(b, 'power_mw_mean') / 1000 if b and b.get('power_mw_mean') else float('nan'))} | "
            f"{f1(g(b, 'energy_mj_per_frame'))} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--image-models", nargs="*", default=None, help="restrict inference tiles to these keys")
    ap.add_argument("--notes", default=os.path.join(ROOT, "docs", "benchmark_notes.json"),
                    help="optional JSON with narrative text per section")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    os.makedirs(IMG, exist_ok=True)
    ms = models()
    notes = jload(args.notes) or {}

    chart_pareto(ms, os.path.join(IMG, "pareto.png"))
    names = list(CLASSES.values())
    hm = [r for r in ms if r["eval"] and not r.get("legacy")]
    chart_heatmap([r["label"] for r in hm], names,
                  [[g(r["eval"], "detection", "per_class", n, "AP50") for n in names] for r in hm],
                  "Box AP50 per class (leakage-free test split)", os.path.join(IMG, "per_class_ap50.png"))
    buckets = ["tiny", "small", "medium", "large"]
    chart_heatmap([r["label"] for r in hm], ["tiny <16 px", "small 16–32", "medium 32–96", "large >96"],
                  [[g(r["eval"], "detection", f"mAP50_{b}") for b in buckets] for r in hm],
                  "Box mAP50 by object size", os.path.join(IMG, "size_ap50.png"))
    chart_latency(ms, os.path.join(IMG, "latency.png"))
    chart_energy(ms, os.path.join(IMG, "energy.png"))
    segs = [r for r in ms if r["task"] == "seg" and r["val"]]
    if segs:
        cls6 = list(CLASSES.values())
        chart_heatmap([r["label"] for r in segs], cls6,
                      [[g(r["val"], "mask_per_class_map50", n) for n in cls6] for r in segs],
                      "Mask AP50 per class", os.path.join(IMG, "seg_per_class.png"))
    if args.no_images:
        uc = {k: os.path.join(IMG, f"uc_{k}.jpg") for k, *_ in USE_CASES if os.path.exists(os.path.join(IMG, f"uc_{k}.jpg"))}
        counts = {}
    else:
        uc, counts = use_case_images(ms, only=args.image_models)

    rel = lambda p: os.path.relpath(p, os.path.dirname(OUT_MD))  # noqa: E731
    md = [
        "# RESAM competitor benchmark — Jetson Orin Nano", "",
        notes.get("intro", ""), "",
        status_section(), "",
        "## Protocol", "",
        "* **Same data for every model**: `MasterSplits/splits_blocked.json` — leakage-free blocked-temporal split "
        "(372 train / 64 val / 80 test frames, 0 test frames with a train neighbour within 31 s), radar-ring ROI crop, "
        "colour input, original 6 classes.",
        "* **Same training recipe** (`tools/train_yolo.py`): COCO-pretrained weights, 960 px, 40 epochs (patience 12), "
        "batch 2, identical augmentations, seed 0, trained **on this Jetson** under a memory watchdog.",
        "* Segmentation models train on `YOLO_Dataset_v2_seg`: labelled polygons + echo-shaped pseudo-masks for "
        "box-only labels (`--refine-boxes`).",
        "* **Accuracy**: box mAP50 from `tools/evaluate.py` (same code for every model incl. non-YOLO; COCO 101-pt, "
        "full-frame coordinates) and, for ultralytics models, ultralytics `val` on the same test images. "
        "**Segmentation headline = instance mask mAP50**: the objects are small, countable instances (a 15 px echo "
        "barely moves pixel IoU while a coastline dominates it), so per-instance mask AP is the metric that reflects "
        "what matters; semantic mIoU is reported alongside for comparison with Mask2Former.",
        "* **Speed**: TensorRT FP16 engines built on the device (INT8-mixed where noted), batch 1, 960×960, 150 "
        "iterations on real 1280×1024 frames; FPS / latency are end-to-end (crop + upload + inference + decode). "
        "Power = whole-board VDD_IN from jtop (idle 5.7 W), 15 W mode, `jetson_clocks` off.",
        "* † Mask2Former is the notebook's existing checkpoint (trained on the legacy random split, cannot be "
        "retrained on an 8 GB Nano): evaluated on its own legacy test set, colour input — reference only.", "",
        "## Leaderboard — detection", "",
        "![Accuracy vs speed](" + rel(os.path.join(IMG, "pareto.png")) + ")", "",
        det_table(ms), "",
        notes.get("detection", ""), "",
        "## Leaderboard — segmentation", "",
        seg_table(ms), "",
        ("![Mask AP50 per class](" + rel(os.path.join(IMG, "seg_per_class.png")) + ")") if segs else "", "",
        notes.get("segmentation", ""), "",
        "## Where the accuracy goes", "",
        "![Per-class AP50](" + rel(os.path.join(IMG, "per_class_ap50.png")) + ")", "",
        "![AP50 by size](" + rel(os.path.join(IMG, "size_ap50.png")) + ")", "",
        notes.get("breakdown", ""), "",
        "## Runtime on the board", "",
        "![Latency breakdown](" + rel(os.path.join(IMG, "latency.png")) + ")", "",
        "![Energy per frame](" + rel(os.path.join(IMG, "energy.png")) + ")", "",
        notes.get("runtime", ""), "",
        "## Inference examples by use case", "",
        "Test-split frames (never seen in training). Each tile: the radar ROI with the model's detections "
        "(conf ≥ 0.25, class colours as in the legend) over the ground-truth boxes (thin white); the first tile "
        "shows the ground truth alone. Latency in the tile header is a single-frame measurement.", "",
    ]
    for key, title, base, desc in USE_CASES:
        md += [f"### {title}", "", f"`{base}` — {desc}", ""]
        if key in uc:
            md += [f"![{title}]({rel(uc[key])})", ""]
        if notes.get(f"uc_{key}"):
            md += [notes[f"uc_{key}"], ""]
    shots = [("coast_orange", "Coastal approach, orange console"), ("clutter_green", "Sea clutter, green console"),
             ("traffic_orange", "Traffic, orange console"), ("open_sea_orange", "Open sea, 6 NM")]
    shots = [(k, t) for k, t in shots if os.path.exists(os.path.join(ROOT, "docs", "img", "dashboard", f"dashboard_{k}.png"))]
    if shots:
        md += ["## Live dashboard on this Jetson", "",
               "`tools/dashboard.py` running YOLO26s INT8-mixed on the Orin Nano, replaying **test-split** frames at "
               "1 fps (`tools/dashboard_screenshots.sh`): annotated live frame, runtime tiles (throughput, latency "
               "percentiles, board power, GPU, RAM), latency / power / detection-count history, per-class counts, "
               "sweep-to-sweep tracks and running accuracy against the labels next to the frames.", ""]
        for k, t in shots:
            md += [f"### {t}", "", f"![{t}](img/dashboard/dashboard_{k}.png)", ""]
        if notes.get("dashboard"):
            md += [notes["dashboard"], ""]
    md += ["## Recommendation", "", notes.get("recommendation", ""), "",
           "## Reproduce", "", "```bash",
           ".venv/bin/python tools/run_competitors.py            # train + val + TensorRT + benchmark + eval (resumable)",
           ".venv/bin/python tools/make_benchmark_report.py      # this file, charts and images", "```", ""]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))
    json.dump(counts, open(os.path.join(COMP, "use_case_counts.json"), "w"), indent=2)
    print("->", OUT_MD)


if __name__ == "__main__":
    main()
