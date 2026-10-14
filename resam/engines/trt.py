"""TensorRT runner for YOLO26 / YOLO11 detection and segmentation engines (TensorRT 8.5 API,
JetPack 5.1.x). GPU buffers are torch tensors, so no pycuda context juggling is needed.

Supported output layouts
  end-to-end det  : (1, K, 6)        x1 y1 x2 y2 score cls      (YOLO26 default, NMS-free)
  end-to-end seg  : (1, K, 6+nm) + protos (1, nm, mh, mw)
  raw det head    : (1, 4+nc, A)     cx cy w h + class scores   (YOLO11/v8, NMS here)
  raw seg head    : (1, 4+nc+nm, A) + protos
Engines written by ultralytics carry a JSON metadata header (names, imgsz) which is read.
"""

import json

import numpy as np

from . import Preprocessor, Result, Timer


# explicit map: trt.nptype() uses np.bool, removed in numpy >= 1.24 (TensorRT 8.5 + numpy 1.24)
_TORCH_DTYPES = {}


def _init_dtypes():
    import torch
    _TORCH_DTYPES.update({"DataType.FLOAT": torch.float32, "DataType.HALF": torch.float16,
                          "DataType.INT32": torch.int32, "DataType.INT8": torch.int8,
                          "DataType.BOOL": torch.bool, "DataType.UINT8": torch.uint8})


def load_engine_bytes(path):
    with open(path, "rb") as f:
        data = f.read()
    meta = {}
    n = int.from_bytes(data[:4], byteorder="little", signed=True)
    if 0 < n < 100000:
        try:
            meta = json.loads(data[4:4 + n].decode("utf-8"))
            data = data[4 + n:]
        except (UnicodeDecodeError, json.JSONDecodeError):
            meta = {}
    return data, meta


class TRTYoloEngine:
    name = "tensorrt"

    def __init__(self, weights, imgsz=None, crop="legacy", conf=0.25, iou=0.6, class_ids=None,
                 class_names=None, mask_thr=0.5, pad_small=False, **_):
        import tensorrt as trt
        import torch

        self.torch = torch
        _init_dtypes()
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        data, meta = load_engine_bytes(weights)
        self.meta = meta
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {weights} (built with another TensorRT version?)")
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.conf, self.iou, self.mask_thr = conf, iou, mask_thr

        self.buffers, self.inputs, self.outputs = [], [], []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = _TORCH_DTYPES[str(self.engine.get_binding_dtype(i))]
            shape = tuple(self.engine.get_binding_shape(i))
            if self.engine.binding_is_input(i):
                if -1 in shape:
                    s = int(imgsz or meta.get("imgsz", [640])[0])
                    shape = (1, 3, s, s)
                    self.ctx.set_binding_shape(i, shape)
                self.inputs.append((i, name, shape, dtype))
            buf = None
            self.buffers.append([name, shape, dtype, buf])
        for i, (name, shape, dtype, _) in enumerate(self.buffers):
            shape = tuple(self.ctx.get_binding_shape(i))
            t = torch.empty(shape, dtype=dtype, device="cuda")
            self.buffers[i] = [name, shape, dtype, t]
            if not self.engine.binding_is_input(i):
                self.outputs.append(i)
        self.bindings = [int(b[3].data_ptr()) for b in self.buffers]
        in_shape = self.inputs[0][2]
        self.imgsz = int(in_shape[-1])
        self.in_dtype = self.inputs[0][3]
        self.pre = Preprocessor(self.imgsz, crop=crop, rgb=True, pad_small=bool(pad_small))
        self._pinned = None

        names = meta.get("names") or class_names or {}
        names = {int(k): v for k, v in names.items()}
        self.class_ids = class_ids or {k: k + 1 for k in names} or None
        self.class_names = {self.class_ids[k]: v for k, v in names.items()} if names else {}

        self.proto_idx = next((i for i in self.outputs if len(self.buffers[i][1]) == 4), None)
        self.pred_idx = next(i for i in self.outputs if i != self.proto_idx)
        ps = self.buffers[self.pred_idx][1]
        self.nm = self.buffers[self.proto_idx][1][1] if self.proto_idx is not None else 0
        # end-to-end YOLO26: (1, K<=1000, 6+nm) xyxy,score,cls | RT-DETR: (1, 300, 4+nc) normalised
        # cxcywh + per-class sigmoid scores | raw YOLO11/v8 heads: (1, 4+nc+nm, anchors)
        queries_first = ps[1] > ps[2]
        self.end2end = queries_first and ps[-1] == 6 + self.nm
        self.rtdetr = queries_first and not self.end2end and self.nm == 0

    def __call__(self, frame):
        torch = self.torch
        t = Timer()
        sync = self.stream.synchronize
        t.start("pre")
        img, info = self.pre(frame)
        if self._pinned is None or self._pinned.shape != img.shape:
            self._pinned = torch.empty(img.shape, dtype=torch.uint8).pin_memory()
        self._pinned.numpy()[...] = img  # pinned staging -> true async H2D copy
        with torch.cuda.stream(self.stream):
            x = self._pinned.to("cuda", non_blocking=True)
            x = x.permute(2, 0, 1).unsqueeze(0).to(self.in_dtype).div_(255.0)
            self.buffers[self.inputs[0][0]][3].copy_(x)
        t.stop(sync)
        t.start("infer")
        self.ctx.execute_async_v2(self.bindings, self.stream.cuda_stream)
        t.stop(sync)
        t.start("post")
        res = self._decode(info, frame.shape)
        t.stop()
        res.timings = t.t
        return res

    # ------------------------------------------------------------------ decoding
    def _decode(self, info, frame_shape):
        torch = self.torch
        pred = self.buffers[self.pred_idx][3][0].float()
        protos = self.buffers[self.proto_idx][3][0].float() if self.proto_idx is not None else None
        if self.rtdetr:
            sc, cls = pred[:, 4:].max(1)
            keep = sc >= self.conf
            p, scores, cls = pred[keep], sc[keep], cls[keep]
            xy, wh = p[:, :2] * self.imgsz, p[:, 2:4] * self.imgsz / 2
            boxes = torch.cat([xy - wh, xy + wh], 1)
            coefs = None
        elif self.end2end:
            keep = pred[:, 4] >= self.conf
            p = pred[keep]
            boxes, scores, cls = p[:, :4], p[:, 4], p[:, 5].long()
            coefs = p[:, 6:6 + self.nm] if self.nm else None
        else:
            p = pred.transpose(0, 1)  # (A, 4+nc+nm)
            nc = p.shape[1] - 4 - self.nm
            sc, cls = p[:, 4:4 + nc].max(1)
            keep = sc >= self.conf
            p, sc, cls = p[keep], sc[keep], cls[keep]
            xy, wh = p[:, :2], p[:, 2:4] / 2
            boxes = torch.cat([xy - wh, xy + wh], 1)
            import torchvision
            k = torchvision.ops.batched_nms(boxes, sc, cls, self.iou)[:300]
            boxes, scores, cls = boxes[k], sc[k], cls[k]
            coefs = p[k, 4 + nc:] if self.nm else None
        boxes = boxes.clamp(0, self.imgsz)
        cls_np = cls.cpu().numpy()
        ids = np.array([self.class_ids.get(int(c), int(c) + 1) if self.class_ids else int(c) + 1
                        for c in cls_np], np.int32)
        res = Result(Preprocessor.boxes_to_frame(boxes.cpu().numpy(), info),
                     scores.cpu().numpy().astype(np.float32), ids)
        if protos is not None and len(ids):
            nm, mh, mw = protos.shape
            masks = torch.sigmoid(coefs @ protos.view(nm, -1)).view(-1, mh, mw)
            r = torch.tensor([mw / self.imgsz, mh / self.imgsz, mw / self.imgsz, mh / self.imgsz], device=masks.device)
            b = boxes * r
            xs = torch.arange(mw, device=masks.device)[None, None, :]
            ys = torch.arange(mh, device=masks.device)[None, :, None]
            inside = ((xs >= b[:, 0, None, None]) & (xs < b[:, 2, None, None]) &
                      (ys >= b[:, 1, None, None]) & (ys < b[:, 3, None, None]))
            masks = (masks > self.mask_thr) & inside
            order = torch.argsort(scores)  # paint low score first
            sem = torch.zeros((mh, mw), dtype=torch.uint8, device=masks.device)
            idt = torch.as_tensor(ids, device=masks.device, dtype=torch.uint8)
            for i in order.tolist():
                sem[masks[i]] = idt[i]
            sem = torch.nn.functional.interpolate(sem[None, None].float(), size=(self.imgsz, self.imgsz),
                                                  mode="nearest")[0, 0].byte().cpu().numpy()
            res.semantic = Preprocessor.semantic_to_frame(sem, info, frame_shape)
        return res
