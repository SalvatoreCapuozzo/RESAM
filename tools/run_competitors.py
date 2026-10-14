#!/usr/bin/env python3
"""Resumable competitor pipeline: train -> ultralytics val (test) -> TensorRT FP16 -> benchmark
-> RESAM evaluation, for every model in JOBS, one at a time (the Orin Nano cannot overlap them).

Every step is skipped when its output exists, so after an OOM/reboot just run it again:
  nohup .venv/bin/python tools/run_competitors.py > models/competitors.log 2>&1 &
Each child process is killed if MemAvailable drops below --min-mem-mb (protects the board).
"""

import argparse
import json
import os
import signal
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
TRAIN = os.path.join(ROOT, "models", "train")
ENG = os.path.join(ROOT, "models", "engines")
RES = os.path.join(ROOT, "results", "competitors")
DET = os.path.join(ROOT, "YOLO_Dataset_v2", "data.yaml")
SEG = os.path.join(ROOT, "YOLO_Dataset_v2_seg", "data.yaml")

# name, init weights / cfg, dataset, task, extra args
JOBS = [
    ("yolo26s", "yolo26s.pt", DET, "detect", {}),            # reference (already trained)
    ("yolo26n", "yolo26n.pt", DET, "detect", {}),
    ("yolo11s", "yolo11s.pt", DET, "detect", {}),
    ("yolov8s", "yolov8s.pt", DET, "detect", {}),
    ("yolo26s-seg", "yolo26s-seg.pt", SEG, "segment", {}),
    ("yolo11s-seg", "yolo11s-seg.pt", SEG, "segment", {}),
    ("yolo26s-p2", "yolo26s-p2.yaml", DET, "detect", {"init": "yolo26s.pt"}),
    ("rtdetr-l", "rtdetr-l.pt", DET, "detect", {}),
]
RUN_NAME = {"yolo26s": "yolo26s_roi960_blocked"}


def mem_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // 1024
    return 1 << 30


def run(cmd, log, min_mem, cwd=ROOT):
    print(f"$ {' '.join(cmd)}", flush=True)
    with open(log, "a") as lf:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        killed = {"v": False}

        def guard():
            low = 0
            while p.poll() is None:
                low = low + 1 if mem_available_mb() < min_mem else 0
                if low >= 3:  # ~6 s below the floor
                    killed["v"] = True
                    os.killpg(p.pid, signal.SIGKILL)
                    return
                time.sleep(2)

        threading.Thread(target=guard, daemon=True).start()
        rc = p.wait()
    if killed["v"]:
        print(f"   !! killed: MemAvailable < {min_mem} MB", flush=True)
        return "oom"
    print(f"   rc={rc}", flush=True)
    return "ok" if rc == 0 else f"rc{rc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--min-mem-mb", type=int, default=250)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(ENG, exist_ok=True)
    status_path = os.path.join(RES, "status.json")
    status = json.load(open(status_path)) if os.path.exists(status_path) else {}

    for attempt in (1, 2):
        if attempt == 2:
            retry = [n for n, *_ in JOBS if any(v == "oom" for v in status.get(n, {}).values())]
            if not retry:
                break
            print(f"\n##### retry pass for {retry}", flush=True)
            for n in retry:
                status[n] = {k: v for k, v in status[n].items() if v == "ok"}
        run_jobs(args, status, status_path, attempt)
    print("ALL_DONE", flush=True)


def run_jobs(args, status, status_path, attempt):
    for name, model, data, task, extra in JOBS:
        if args.only and name not in args.only:
            continue
        if attempt == 2 and not any(v != "ok" for v in status.get(name, {"x": "pending"}).values()) \
                and os.path.exists(os.path.join(RES, f"{name}_eval_visual.json")):
            continue
        st = status.setdefault(name, {})
        run_name = RUN_NAME.get(name, f"{name}_roi960_blocked")
        best = os.path.join(TRAIN, run_name, "weights", "best.pt")
        log = os.path.join(RES, f"{name}.log")
        print(f"\n===== {name} =====", flush=True)

        last = os.path.join(TRAIN, run_name, "weights", "last.pt")
        done = name in RUN_NAME or st.get("train") == "ok"  # best.pt alone may be a killed partial run
        if not done:
            cmd = [PY, os.path.join(ROOT, "tools", "train_yolo.py"), "--model", model, "--data", data, "--name", run_name]
            if extra.get("init"):
                cmd += ["--init", extra["init"]]
            if os.path.exists(last):
                cmd += ["--resume", last]
            st["train"] = run(cmd, log, args.min_mem_mb, cwd=TRAIN)
            json.dump(status, open(status_path, "w"), indent=2)
            if st["train"] != "ok" or not os.path.exists(best):
                print(f"   training failed ({st['train']}), skipping {name}", flush=True)
                continue

        val_json = os.path.join(RES, f"{name}_ultra_val.json")
        if not os.path.exists(val_json):
            code = (
                "import json,os;os.environ['YOLO_AUTOINSTALL']='false'\n"
                "from ultralytics import YOLO, RTDETR\n"
                f"M=RTDETR if 'rtdetr' in {name!r} else YOLO\n"
                f"m=M({best!r});r=m.val(data={data!r},split='test',imgsz=960,batch=2,workers=0,plots=False,half=True)\n"
                "out={'box_map50':float(r.box.map50),'box_map50_95':float(r.box.map),"
                "'box_per_class_map50':{m.names[i]:float(v) for i,v in enumerate(r.box.ap50)},"
                "'params_M':sum(p.numel() for p in m.model.parameters())/1e6}\n"
                "if hasattr(r,'seg'):\n"
                "    out.update({'mask_map50':float(r.seg.map50),'mask_map50_95':float(r.seg.map),"
                "'mask_per_class_map50':{m.names[i]:float(v) for i,v in enumerate(r.seg.ap50)}})\n"
                f"json.dump(out,open({val_json!r},'w'),indent=2)\n"
            )
            st["val"] = run([PY, "-W", "ignore", "-c", code], log, args.min_mem_mb)

        engine = os.path.join(ENG, f"{name}_roi960_fp16.engine")
        if not os.path.exists(engine):
            local = os.path.join(ENG, f"{name}_roi960_blocked.pt")
            if not os.path.exists(local):
                subprocess.run(["cp", best, local], check=True)
            st["export"] = run([PY, os.path.join(ROOT, "tools", "export_trt.py"), "--weights", local,
                                "--imgsz", "960", "--precision", "fp16", "--workspace", "1", "--out", engine], log, args.min_mem_mb, cwd=ENG)
            if not os.path.exists(engine):
                print("   export failed", flush=True)
                json.dump(status, open(status_path, "w"), indent=2)
                continue

        spec = f"trt:{os.path.relpath(engine, ROOT)}@crop=roi"
        bench = os.path.join(RES, f"{name}_bench.json")
        if not os.path.exists(bench):
            st["bench"] = run([PY, "-W", "ignore", os.path.join(ROOT, "tools", "benchmark.py"), "--iters", "150",
                               "--engine", spec, "--out", bench], log, args.min_mem_mb)
        for mapping in ("original", "visual"):
            ev = os.path.join(RES, f"{name}_eval_{mapping}.json")
            if not os.path.exists(ev):
                st[f"eval_{mapping}"] = run([PY, "-W", "ignore", os.path.join(ROOT, "tools", "evaluate.py"),
                                             "--engine", spec, "--mapping", mapping, "--out", ev], log, args.min_mem_mb)
        json.dump(status, open(status_path, "w"), indent=2)
        print(f"   done {name}: {st}", flush=True)


if __name__ == "__main__":
    main()
