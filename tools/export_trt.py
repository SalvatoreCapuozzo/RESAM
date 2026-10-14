#!/usr/bin/env python3
"""Export YOLO26 / YOLO11 (det or seg) to a TensorRT engine for this Jetson.

  # trained weights -> FP16 engine at 960 (radar ROI crop is 958 px on the 1280x1024 display)
  python tools/export_trt.py --weights best.pt --imgsz 960 --precision fp16

  # INT8 with calibration on real radar frames (same preprocessing as inference)
  python tools/export_trt.py --weights best.pt --imgsz 960 --precision int8 \
      --calib YOLO_Dataset_v2/images/train --calib-n 200

  # architecture-only speed test (random weights): any ultralytics model yaml
  python tools/export_trt.py --weights yolo26s.yaml --nc 6 --imgsz 960 --precision fp16

Engines are written with the same JSON metadata header ultralytics uses (names, imgsz,
task) plus RESAM fields (crop mode, class-id mapping), so resam.engines.trt and the
ultralytics runtime can both load them. TensorRT engines are specific to this
GPU + TensorRT version (Orin Nano, TRT 8.5.2): build them on the device.
"""

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never let ultralytics pip-install on the device
os.environ.setdefault("YOLO_OFFLINE", "true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def export_onnx(weights, imgsz, nc=None, opset=17):
    from ultralytics import YOLO
    if weights.endswith(".yaml"):
        import yaml
        from ultralytics.nn.tasks import yaml_model_load
        d = yaml_model_load(weights)  # resolves 'yolo26s-seg.yaml' -> family yaml + scale 's'
        if nc:
            d["nc"] = nc
        d.pop("yaml_file", None)
        cfg = os.path.join("/tmp", "resam_" + os.path.basename(weights))
        with open(cfg, "w") as f:
            yaml.safe_dump(d, f)
        model = YOLO(cfg, task="segment" if "-seg" in weights else "detect")
    else:
        model = YOLO(weights)
    out = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False, batch=1)
    names = {int(k): v for k, v in model.names.items()}
    return out, names, model.task


class ImageCalibrator:
    """IInt8EntropyCalibrator2 fed with radar frames (already ROI-cropped images, e.g. the
    YOLO_Dataset_v2 train split), resized to the engine input."""

    def __init__(self, files, imgsz, cache, kind="entropy"):
        import tensorrt as trt
        import torch

        self.torch = torch
        self.files = files
        self.imgsz = imgsz
        self.cache = cache
        self.i = 0
        self.buf = torch.empty((1, 3, imgsz, imgsz), dtype=torch.float32, device="cuda")
        outer = self

        base = trt.IInt8MinMaxCalibrator if kind == "minmax" else trt.IInt8EntropyCalibrator2

        class Calib(base):
            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                return outer.next_batch()

            def read_calibration_cache(self):
                return open(outer.cache, "rb").read() if os.path.exists(outer.cache) else None

            def write_calibration_cache(self, c):
                with open(outer.cache, "wb") as f:
                    f.write(c)

        self.calibrator = Calib()

    def next_batch(self):
        import cv2
        if self.i >= len(self.files):
            return None
        img = cv2.imread(self.files[self.i])
        self.i += 1
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.torch.from_numpy(img).cuda().permute(2, 0, 1).unsqueeze(0).float().div(255)
        self.buf.copy_(x)
        return [int(self.buf.data_ptr())]


def build_engine(onnx_path, out_path, precision, workspace_gb, calib=None, meta=None, verbose=False,
                 fp16_layers=None):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if precision in ("fp16", "int8"):
        cfg.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = calib.calibrator
        if fp16_layers:
            # Mixed precision: pure INT8 with entropy calibration wiped out tiny/small-object
            # recall (test mAP50 0.50 -> 0.13). Keep the detection head in FP16.
            import re
            pat = re.compile(fp16_layers)
            n = 0
            for i in range(net.num_layers):
                layer = net.get_layer(i)
                outs = [layer.get_output(o) for o in range(layer.num_outputs)]
                # only floating-point compute layers: int constants / TopK indices / shape ops can't be FP16
                if (pat.search(layer.name) and layer.type != trt.LayerType.CONSTANT
                        and outs and all(t.dtype == trt.float32 for t in outs)):
                    layer.precision = trt.float16
                    for o in range(layer.num_outputs):
                        layer.set_output_type(o, trt.float16)
                    n += 1
            cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            print(f"pinned {n}/{net.num_layers} layers matching '{fp16_layers}' to FP16")
    t0 = time.time()
    ser = builder.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError("engine build failed")
    m = json.dumps(meta or {}).encode()
    with open(out_path, "wb") as f:
        f.write(len(m).to_bytes(4, byteorder="little", signed=True))
        f.write(m)
        f.write(ser)
    print(f"built {out_path} in {time.time() - t0:.0f}s ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help=".pt, .onnx or model .yaml (random init)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    ap.add_argument("--nc", type=int, default=6, help="classes for .yaml models")
    ap.add_argument("--calib", default="YOLO_Dataset_v2/images/train")
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--calibrator", choices=["entropy", "minmax"], default="minmax")
    ap.add_argument("--fp16-layers", default=r"/model\.23/",
                    help="(int8) regex of layer names kept in FP16; '' = pure INT8")
    ap.add_argument("--workspace", type=float, default=2.0, help="GB (Orin Nano shares 8 GB with the CPU)")
    ap.add_argument("--crop", default="roi", choices=["roi", "legacy"], help="preprocessing the model was trained with")
    ap.add_argument("--out", default=None)
    ap.add_argument("--onnx-only", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    names, task = {}, "detect"
    if args.onnx_only:
        onnx_path, names, task = export_onnx(args.weights, args.imgsz, args.nc)
        print("@@" + json.dumps({"onnx": onnx_path, "names": names, "task": task}))
        return
    if args.weights.endswith(".onnx"):
        onnx_path = args.weights
    else:
        # ONNX export in a child process: the export leaves enough GPU/host memory behind on
        # the 8 GB shared-memory Orin Nano that trt.Builder() then fails to initialise.
        import subprocess
        cmd = [sys.executable, os.path.abspath(__file__), "--onnx-only", "--weights", args.weights,
               "--imgsz", str(args.imgsz), "--nc", str(args.nc)]
        out_lines = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True).stdout.splitlines()
        info = json.loads([ln for ln in out_lines if ln.startswith("@@")][-1][2:])
        onnx_path, names, task = info["onnx"], info["names"], info["task"]
    out = args.out or os.path.splitext(onnx_path)[0] + f"_{args.precision}_{args.imgsz}.engine"
    calib = None
    if args.precision == "int8":
        files = sorted(glob.glob(os.path.join(args.calib, "*.png")) + glob.glob(os.path.join(args.calib, "*.jpg")))
        files = files[:: max(1, len(files) // args.calib_n)][: args.calib_n]
        if not files:
            raise SystemExit(f"no calibration images in {args.calib}")
        calib = ImageCalibrator(files, args.imgsz, os.path.splitext(out)[0] + f".{args.calibrator}.calib",
                                kind=args.calibrator)
    meta = {"names": names, "imgsz": [args.imgsz, args.imgsz], "task": task, "batch": 1,
            "resam": {"crop": args.crop, "precision": args.precision, "source": os.path.basename(args.weights)}}
    if args.precision == "int8":
        meta["resam"].update({"calibrator": args.calibrator, "fp16_layers": args.fp16_layers})
    build_engine(onnx_path, out, args.precision, args.workspace, calib, meta,
                 fp16_layers=args.fp16_layers if args.precision == "int8" else None)


if __name__ == "__main__":
    main()
