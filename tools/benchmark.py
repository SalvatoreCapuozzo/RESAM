#!/usr/bin/env python3
"""On-device benchmark: latency percentiles per stage, throughput, board power and
energy per frame, for any mix of backends, on real radar frames.

  python tools/benchmark.py --frames "IMMAGINI/Cartour delta 1.30/202109150130" \
      --engine trt:models/yolo26s_fp16_960.engine \
      --engine trt:models/yolo26s_int8_960.engine \
      --engine mask2former:vit_wradar_fold_1_20260414_152413.pth \
      --engine classical --iters 100 --out results/benchmark.json

Engine spec: kind[:weights][@key=value,...], e.g. ultralytics:best.pt@imgsz=1024,crop=legacy
Power comes from the jtop service (VDD_IN, total board input) - no sudo needed.
For stable numbers run `sudo jetson_clocks` first (not done automatically).
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resam.engines import create_engine  # noqa: E402
from resam.metrics import LatencyStats  # noqa: E402
from resam.sources import FolderSource  # noqa: E402
from resam.telemetry import Telemetry  # noqa: E402


def parse_spec(spec):
    kw = {}
    if "@" in spec:
        spec, opts = spec.split("@", 1)
        for kv in opts.split(","):
            k, v = kv.split("=")
            kw[k] = int(v) if v.isdigit() else v
    kind, _, weights = spec.partition(":")
    if weights:
        kw["weights"] = weights
    return kind, kw


def run_one(spec, frames, iters, warmup, tel):
    kind, kw = parse_spec(spec)
    t0 = time.time()
    eng = create_engine(kind, **kw)
    load_s = time.time() - t0
    for i in range(warmup):
        eng(frames[i % len(frames)])
    stats = LatencyStats()
    time.sleep(1.5)  # let the power reading settle onto this workload
    e0, p_samples = tel.energy_j, []
    t_start = time.perf_counter()
    n_det = []
    last_ts = None
    for i in range(iters):
        f = frames[i % len(frames)]
        t = time.perf_counter()
        r = eng(f)
        total = (time.perf_counter() - t) * 1000
        stats.add(total=total, **r.timings)
        n_det.append(len(r.boxes))
        snap = tel.latest
        p, ts = snap.get("power_mw"), snap.get("ts")
        if p and ts != last_ts and (ts is None or time.time() - ts < 2.0):  # fresh samples only
            p_samples.append(p)
            last_ts = ts
    wall = time.perf_counter() - t_start
    if wall < 5:  # too short to sample power at 2 Hz: keep running the workload
        while time.perf_counter() - t_start < 5:
            eng(frames[0])
            snap = tel.latest
            if snap.get("power_mw") and snap.get("ts") != last_ts:
                p_samples.append(snap["power_mw"])
                last_ts = snap.get("ts")
    s = stats.summary()
    out = {
        "engine": spec, "kind": kind, "load_s": load_s, "iters": iters,
        "throughput_fps": iters / wall,
        "latency_ms": {k: v for k, v in s.items() if isinstance(v, dict)},
        "power_mw_mean": float(np.mean(p_samples)) if p_samples else None,
        "energy_mj_per_frame": (float(np.mean(p_samples)) * wall / iters) if p_samples else None,
        "energy_j_integrated": tel.energy_j - e0,
        "dets_per_frame": float(np.mean(n_det)),
        "telemetry_end": {k: v for k, v in tel.snapshot().items() if k in ("gpu_pct", "ram_used_mb", "temp_gpu", "nvp_model", "jetson_clocks")},
    }
    try:
        import torch
        if torch.cuda.is_available():
            out["torch_max_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    del eng
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    return out


def to_markdown(results, idle_mw=None):
    rows = ["| engine | FPS | total p50 / p95 ms | infer p50 ms | pre / post p50 ms | power W | mJ / frame |",
            "|---|---|---|---|---|---|---|"]
    for r in results:
        L = r["latency_ms"]
        g = lambda k, q="p50": L.get(k, {}).get(q, float("nan"))  # noqa: E731
        pw = r["power_mw_mean"]
        rows.append(f"| {r['engine']} | {r['throughput_fps']:.1f} | {g('total'):.1f} / {g('total', 'p95'):.1f} | "
                    f"{g('infer'):.1f} | {g('pre'):.1f} / {g('post'):.1f} | "
                    f"{(pw / 1000 if pw else float('nan')):.2f} | {(r['energy_mj_per_frame'] or float('nan')):.0f} |")
    if idle_mw:
        rows.append(f"\nIdle board power: {idle_mw / 1000:.2f} W")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="IMMAGINI/Cartour delta 1.30/202109150130")
    ap.add_argument("--engine", action="append", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()

    frames = [f for f, _ in zip((x for x, _ in FolderSource(args.frames)), range(args.n_frames))]
    tel = Telemetry(interval=0.5)
    time.sleep(3)
    idle = tel.latest.get("power_mw")
    results = []
    for spec in args.engine:
        print(f"== {spec}", flush=True)
        try:
            r = run_one(spec, frames, args.iters, args.warmup, tel)
        except Exception as e:  # keep benchmarking the others
            print(f"   FAILED: {type(e).__name__}: {e}")
            results.append({"engine": spec, "error": f"{type(e).__name__}: {e}"})
            continue
        L = r["latency_ms"]["total"]
        print(f"   {r['throughput_fps']:.1f} FPS | p50 {L['p50']:.1f} ms p95 {L['p95']:.1f} ms | "
              f"{(r['power_mw_mean'] or 0) / 1000:.2f} W | {r['energy_mj_per_frame'] or 0:.0f} mJ/frame", flush=True)
        results.append(r)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {"device": "Jetson Orin Nano 8GB", "idle_power_mw": idle, "telemetry": tel.backend,
               "frames": args.frames, "results": results}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    ok = [r for r in results if "error" not in r]
    md = to_markdown(ok, idle)
    with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
        f.write(md + "\n")
    print(md)
    tel.stop()


if __name__ == "__main__":
    main()
