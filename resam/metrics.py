"""Evaluation metrics for radar-screen detection and segmentation, plus runtime stats.

Detection (DetectionEvaluator)
  * COCO-style AP (101-pt) at IoU 0.50:0.95, AP50, AP75, per class
  * size-bucketed AP (tiny < 16 px, small 16-32, medium 32-96, large > 96; sqrt(area))
    - radar targets are mostly 10-40 px, so the tiny/small buckets are the ones that matter
  * P / R / F1 at a confidence threshold and the F1-optimal threshold
  * radar-style Pd (probability of detection) and false alarms per frame, with either
    IoU or centre-hit matching (IoU is very harsh on 10-20 px echoes)
  * class confusion matrix after class-agnostic matching: shows how much error is
    localisation vs. Ship / Unknown / AIS-Lost confusion
  * class-agnostic AP ("is there a target here at all")
Segmentation (SegEvaluator)
  * confusion-matrix pixel metrics: IoU, Dice/F1, precision, recall per class, mIoU,
    frequency-weighted IoU, pixel accuracy, binary foreground IoU
  * boundary F1 (tolerance in px), useful for Coast/Port polygons
  * object-level Pd/FA from connected components of the semantic maps
Runtime (LatencyStats): mean, p50/p90/p95/p99, max, jitter, FPS.
"""

import math
from collections import defaultdict

import cv2
import numpy as np

SIZE_BUCKETS = {"all": (0, 1e9), "tiny": (0, 16), "small": (16, 32), "medium": (32, 96), "large": (96, 1e9)}
IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
REC_THRS = np.linspace(0, 1, 101)


def box_iou(a, b):
    """IoU matrix between xyxy boxes a (N,4) and b (M,4)."""
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    aa = np.prod(a[:, 2:] - a[:, :2], axis=1)
    ab = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def center_hit(pred, gt, pad=2.0):
    """(N,M) matrix: 1 if pred centre lies inside gt box (padded) - echo-friendly matching."""
    pred = np.asarray(pred, np.float32).reshape(-1, 4)
    gt = np.asarray(gt, np.float32).reshape(-1, 4)
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), np.float32)
    c = (pred[:, :2] + pred[:, 2:]) / 2
    inside = ((c[:, None, 0] >= gt[None, :, 0] - pad) & (c[:, None, 0] <= gt[None, :, 2] + pad) &
              (c[:, None, 1] >= gt[None, :, 1] - pad) & (c[:, None, 1] <= gt[None, :, 3] + pad))
    # rank by IoU among hits so the closest box wins
    return inside.astype(np.float32) * (0.5 + 0.5 * box_iou(pred, gt))


def _sqrt_area(boxes):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    return np.sqrt(np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None))


def greedy_match(scores, sim, thr):
    """Greedy (score-ordered) one-to-one matching. Returns det->gt index (-1 if none)."""
    order = np.argsort(-np.asarray(scores))
    taken = np.zeros(sim.shape[1], bool)
    match = -np.ones(sim.shape[0], int)
    for d in order:
        if sim.shape[1] == 0:
            break
        s = np.where(taken, -1, sim[d])
        g = int(s.argmax())
        if s[g] >= thr:
            match[d] = g
            taken[g] = True
    return match


class DetectionEvaluator:
    def __init__(self, class_names, max_dets=300):
        self.class_names = dict(class_names)
        self.max_dets = max_dets
        self.images = []

    def add(self, gt_boxes, gt_classes, pred_boxes, pred_scores, pred_classes, meta=None):
        self.images.append({
            "gb": np.asarray(gt_boxes, np.float32).reshape(-1, 4),
            "gc": np.asarray(gt_classes, np.int32).reshape(-1),
            "pb": np.asarray(pred_boxes, np.float32).reshape(-1, 4),
            "ps": np.asarray(pred_scores, np.float32).reshape(-1),
            "pc": np.asarray(pred_classes, np.int32).reshape(-1),
            "meta": meta or {},
        })

    # ---------------------------------------------------------------- AP
    def _ap_one(self, cls, iou_thr, size, agnostic=False):
        lo, hi = SIZE_BUCKETS[size]
        scores, tps, ignores = [], [], []
        npos = 0
        for im in self.images:
            gsel = np.ones(len(im["gc"]), bool) if agnostic else im["gc"] == cls
            psel = np.ones(len(im["pc"]), bool) if agnostic else im["pc"] == cls
            gb, pb, ps = im["gb"][gsel], im["pb"][psel], im["ps"][psel]
            if len(ps) > self.max_dets:
                keep = np.argsort(-ps)[:self.max_dets]
                pb, ps = pb[keep], ps[keep]
            g_area = _sqrt_area(gb)
            g_ign = (g_area < lo) | (g_area >= hi)
            npos += int((~g_ign).sum())
            if len(ps) == 0:
                continue
            iou = box_iou(pb, gb)
            order = np.argsort(-ps)
            taken = np.zeros(len(gb), bool)
            tp = np.zeros(len(ps), bool)
            ign = np.zeros(len(ps), bool)
            for d in order:
                best, best_g = iou_thr - 1e-9, -1
                for pass_ignored in (False, True):  # prefer real GT, then ignored GT
                    for g in range(len(gb)):
                        if taken[g] or g_ign[g] != pass_ignored or iou[d, g] < best:
                            continue
                        best, best_g = iou[d, g], g
                    if best_g >= 0:
                        break
                if best_g >= 0:
                    taken[best_g] = True
                    tp[d] = not g_ign[best_g]
                    ign[d] = g_ign[best_g]
                else:
                    a = _sqrt_area(pb[d:d + 1])[0]
                    ign[d] = a < lo or a >= hi
            scores.append(ps)
            tps.append(tp)
            ignores.append(ign)
        if npos == 0:
            return float("nan"), None
        if not scores:
            return 0.0, None
        s = np.concatenate(scores)
        t = np.concatenate(tps)
        ig = np.concatenate(ignores)
        o = np.argsort(-s, kind="mergesort")
        s, t, ig = s[o], t[o], ig[o]
        keep = ~ig
        s, t = s[keep], t[keep]
        ctp = np.cumsum(t)
        cfp = np.cumsum(~t)
        rec = ctp / npos
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        prec_env = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
        q = np.zeros(len(REC_THRS))
        idx = np.searchsorted(rec, REC_THRS, side="left")
        valid = idx < len(prec_env)
        q[valid] = prec_env[idx[valid]]
        return float(q.mean()), {"scores": s, "prec": prec, "rec": rec, "npos": npos}

    def ap_table(self):
        out = {"per_class": {}, "class_agnostic": {}}
        for cls, name in self.class_names.items():
            row = {}
            aps = [self._ap_one(cls, t, "all")[0] for t in IOU_THRESHOLDS]
            row["AP50"] = aps[0]
            row["AP75"] = aps[5]
            row["AP50_95"] = float(np.nanmean(aps)) if not all(map(math.isnan, aps)) else float("nan")
            for size in ("tiny", "small", "medium", "large"):
                row[f"AP50_{size}"] = self._ap_one(cls, 0.5, size)[0]
            row["n_gt"] = int(sum((im["gc"] == cls).sum() for im in self.images))
            out["per_class"][name] = row

        def mean_of(key):
            v = [r[key] for r in out["per_class"].values() if not math.isnan(r[key])]
            return float(np.mean(v)) if v else float("nan")

        out["mAP50"] = mean_of("AP50")
        out["mAP75"] = mean_of("AP75")
        out["mAP50_95"] = mean_of("AP50_95")
        for size in ("tiny", "small", "medium", "large"):
            out[f"mAP50_{size}"] = mean_of(f"AP50_{size}")
        ag = [self._ap_one(None, t, "all", agnostic=True)[0] for t in IOU_THRESHOLDS]
        out["class_agnostic"] = {"AP50": ag[0], "AP50_95": float(np.nanmean(ag))}
        return out

    # ---------------------------------------------------------------- operating point
    def operating_point(self, conf=0.25, match="iou", thr=0.5):
        """Per-class and overall P/R/F1, Pd and false alarms per frame at a confidence."""
        per = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for im in self.images:
            k = im["ps"] >= conf
            pb, pc, ps = im["pb"][k], im["pc"][k], im["ps"][k]
            for cls in self.class_names:
                g = im["gb"][im["gc"] == cls]
                p = pb[pc == cls]
                s = ps[pc == cls]
                sim = box_iou(p, g) if match == "iou" else center_hit(p, g)
                m = greedy_match(s, sim, thr if match == "iou" else 1e-6)
                tp = int((m >= 0).sum())
                per[cls]["tp"] += tp
                per[cls]["fp"] += len(p) - tp
                per[cls]["fn"] += len(g) - tp
        n_img = max(len(self.images), 1)
        res = {"conf": conf, "match": match, "per_class": {}}
        T = {"tp": 0, "fp": 0, "fn": 0}
        for cls, name in self.class_names.items():
            d = per[cls]
            for key in T:
                T[key] += d[key]
            res["per_class"][name] = _prf(d, n_img)
        res["overall"] = _prf(T, n_img)
        return res

    def best_f1_conf(self, match="iou", thr=0.5, grid=np.linspace(0.05, 0.95, 19)):
        best = (-1, 0.25)
        for c in grid:
            f1 = self.operating_point(c, match, thr)["overall"]["F1"]
            if f1 > best[0]:
                best = (f1, float(c))
        return {"F1": best[0], "conf": best[1]}

    def confusion(self, conf=0.25, iou_thr=0.5):
        """Rows = GT class (+ 'background'), cols = predicted class (+ 'missed')."""
        ids = list(self.class_names)
        idx = {c: i for i, c in enumerate(ids)}
        n = len(ids)
        cm = np.zeros((n + 1, n + 1), int)
        for im in self.images:
            k = im["ps"] >= conf
            pb, pc, ps = im["pb"][k], im["pc"][k], im["ps"][k]
            m = greedy_match(ps, box_iou(pb, im["gb"]), iou_thr)
            matched_g = set()
            for d, g in enumerate(m):
                if pc[d] not in idx:
                    continue
                if g >= 0 and im["gc"][g] in idx:
                    cm[idx[im["gc"][g]], idx[pc[d]]] += 1
                    matched_g.add(g)
                else:
                    cm[n, idx[pc[d]]] += 1  # false alarm on background
            for g, c in enumerate(im["gc"]):
                if g not in matched_g and c in idx:
                    cm[idx[c], n] += 1  # missed
        labels = [self.class_names[c] for c in ids]
        return {"rows_gt": labels + ["background"], "cols_pred": labels + ["missed"], "matrix": cm.tolist()}

    def summary(self, conf=0.25):
        out = self.ap_table()
        out["at_conf"] = self.operating_point(conf, "iou", 0.5)
        out["at_conf_center_hit"] = self.operating_point(conf, "center")
        out["best_f1"] = self.best_f1_conf()
        out["confusion"] = self.confusion(conf)
        out["n_images"] = len(self.images)
        return out


def _prf(d, n_img):
    tp, fp, fn = d["tp"], d["fp"], d["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "P": p, "R": r, "F1": f1,
            "Pd": r, "FA_per_frame": fp / n_img}


# -------------------------------------------------------------------- segmentation
class SegEvaluator:
    def __init__(self, class_names, boundary_tol=2):
        self.class_names = dict(class_names)
        self.ids = sorted(self.class_names)
        self.n = max(self.ids) + 1
        self.cm = np.zeros((self.n, self.n), np.int64)
        self.boundary_tol = boundary_tol
        self.bf = defaultdict(list)
        self.obj = DetectionEvaluator(class_names)
        self.per_image = []

    def add(self, gt, pred, valid=None, name=None, components=True):
        gt = gt.astype(np.int64)
        pred = pred.astype(np.int64)
        if valid is not None:
            v = valid.astype(bool)
            g, p = gt[v], pred[v]
        else:
            g, p = gt.ravel(), pred.ravel()
        g = np.clip(g, 0, self.n - 1)
        p = np.clip(p, 0, self.n - 1)
        cm = np.bincount(g * self.n + p, minlength=self.n * self.n).reshape(self.n, self.n)
        self.cm += cm
        ious = []
        for c in self.ids:
            tp = cm[c, c]
            un = cm[c, :].sum() + cm[:, c].sum() - tp
            if un:
                ious.append(tp / un)
            gc, pc = gt == c, pred == c
            if gc.any() or pc.any():
                self.bf[c].append(boundary_f1(gc, pc, self.boundary_tol))
        self.per_image.append({"name": name, "mIoU": float(np.mean(ious)) if ious else float("nan")})
        if components:
            gb, gcl = components_to_boxes(gt, self.ids)
            pb, pcl = components_to_boxes(pred, self.ids)
            self.obj.add(gb, gcl, pb, np.ones(len(pb)), pcl)

    def summary(self):
        cm = self.cm
        res = {"per_class": {}}
        ious, freqs = [], []
        total = cm.sum()
        for c in self.ids:
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            if tp + fp + fn == 0:
                res["per_class"][self.class_names[c]] = None
                continue
            iou = tp / (tp + fp + fn)
            res["per_class"][self.class_names[c]] = {
                "IoU": iou,
                "Dice": 2 * tp / (2 * tp + fp + fn),
                "Precision": tp / (tp + fp) if tp + fp else 0.0,
                "Recall": tp / (tp + fn) if tp + fn else 0.0,
                "BoundaryF1": float(np.mean(self.bf[c])) if self.bf[c] else float("nan"),
                "gt_pixels": int(cm[c, :].sum()),
            }
            ious.append(iou)
            freqs.append(cm[c, :].sum())
        res["mIoU"] = float(np.mean(ious)) if ious else float("nan")
        res["fwIoU"] = float(np.sum(np.array(freqs) * np.array(ious)) / max(np.sum(freqs), 1)) if ious else float("nan")
        res["pixel_acc"] = float(np.trace(cm) / max(total, 1))
        fg_tp = cm[1:, 1:].sum()
        fg_un = cm[1:, :].sum() + cm[:, 1:].sum() - fg_tp
        res["fg_IoU"] = float(fg_tp / max(fg_un, 1))
        op = self.obj.operating_point(conf=0.0, match="iou", thr=0.3)
        res["object_level_iou0.3"] = op
        res["object_level_center_hit"] = self.obj.operating_point(conf=0.0, match="center")
        return res


def boundary_f1(gt, pred, tol=2):
    """Boundary F-score between two binary masks with a pixel tolerance."""
    def edge(m):
        m = m.astype(np.uint8)
        return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0
    ge, pe = edge(gt), edge(pred)
    if not ge.any() and not pe.any():
        return 1.0
    if not ge.any() or not pe.any():
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    gd = cv2.dilate(ge.astype(np.uint8), k) > 0
    pd = cv2.dilate(pe.astype(np.uint8), k) > 0
    p = (pe & gd).sum() / pe.sum()
    r = (ge & pd).sum() / ge.sum()
    return float(2 * p * r / (p + r)) if p + r else 0.0


def components_to_boxes(label_map, class_ids, min_area=4):
    boxes, classes = [], []
    for c in class_ids:
        m = (label_map == c).astype(np.uint8)
        if not m.any():
            continue
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a >= min_area:
                boxes.append([x, y, x + w, y + h])
                classes.append(c)
    return np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(classes, np.int32)


# -------------------------------------------------------------------- runtime
class LatencyStats:
    def __init__(self, window=None):
        self.window = window
        self.samples = defaultdict(list)

    def add(self, **stages_ms):
        for k, v in stages_ms.items():
            s = self.samples[k]
            s.append(float(v))
            if self.window and len(s) > self.window:
                del s[0]

    def summary(self):
        out = {}
        for k, v in self.samples.items():
            if not v:
                continue
            a = np.asarray(v)
            out[k] = {"n": len(a), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                      "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)),
                      "p99": float(np.percentile(a, 99)), "max": float(a.max()),
                      "jitter_std": float(a.std())}
        if "total" in out and out["total"]["mean"] > 0:
            out["fps_mean"] = 1000.0 / out["total"]["mean"]
        return out
