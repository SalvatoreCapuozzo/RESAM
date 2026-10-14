"""Lightweight multi-target tracker over radar sweeps.

Why: a single screenshot cannot tell a small ship from a sea-clutter spike - persistence
can. Ships survive many sweeps with smooth motion; clutter flickers. The tracker
  * associates detections sweep-to-sweep (IoU, then centre distance for tiny echoes),
  * confirms a track after `confirm_hits` hits within its life,
  * votes the class over the track history (stabilises Ship / Unknown / AIS-Lost flips),
  * exposes `noise_like` for tracks that never confirm -> can be relabelled as Noise.
Constant-velocity prediction is enough at 1 frame per sweep; no scipy dependency.
"""

from collections import Counter

import numpy as np

from .metrics import box_iou


class Track:
    _next_id = 1

    def __init__(self, box, score, cls, frame_idx):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = np.asarray(box, np.float32)
        self.vel = np.zeros(2, np.float32)
        self.score = float(score)
        self.cls_votes = Counter({int(cls): float(score)})
        self.hits = 1
        self.misses = 0
        self.first = frame_idx
        self.last = frame_idx
        self.history = [self.center]

    @property
    def center(self):
        return (self.box[:2] + self.box[2:]) / 2

    @property
    def cls(self):
        return self.cls_votes.most_common(1)[0][0]

    def predict(self):
        b = self.box.copy()
        b[:2] += self.vel
        b[2:] += self.vel
        return b

    def update(self, box, score, cls, frame_idx):
        box = np.asarray(box, np.float32)
        c_new = (box[:2] + box[2:]) / 2
        self.vel = 0.6 * self.vel + 0.4 * (c_new - self.center)
        self.box = box
        self.score = 0.7 * self.score + 0.3 * float(score)
        self.cls_votes[int(cls)] += float(score)
        self.hits += 1
        self.misses = 0
        self.last = frame_idx
        self.history.append(c_new)
        if len(self.history) > 50:
            self.history.pop(0)


class SweepTracker:
    def __init__(self, iou_thr=0.2, max_center_dist=25.0, max_misses=3, confirm_hits=3,
                 track_classes=(1, 2, 3, 6), noise_class=2):
        self.iou_thr = iou_thr
        self.max_center_dist = max_center_dist
        self.max_misses = max_misses
        self.confirm_hits = confirm_hits
        self.track_classes = set(track_classes)
        self.noise_class = noise_class
        self.tracks = []
        self.frame_idx = 0
        self.stats = {"created": 0, "confirmed": 0, "expired_unconfirmed": 0}

    def update(self, boxes, scores, classes):
        self.frame_idx += 1
        sel = np.array([int(c) in self.track_classes for c in classes], bool) if len(classes) else np.zeros(0, bool)
        boxes, scores, classes = boxes[sel], scores[sel], classes[sel]
        preds = np.array([t.predict() for t in self.tracks], np.float32).reshape(-1, 4)
        assigned_t, assigned_d = set(), set()
        if len(preds) and len(boxes):
            iou = box_iou(boxes, preds)
            pc = (preds[:, :2] + preds[:, 2:]) / 2
            dc = (boxes[:, :2] + boxes[:, 2:]) / 2
            dist = np.linalg.norm(dc[:, None] - pc[None], axis=2)
            # combined affinity: IoU, or proximity for tiny echoes with no overlap
            aff = np.where(iou >= self.iou_thr, 1.0 + iou, np.clip(1.0 - dist / self.max_center_dist, 0, 1))
            for d, t in sorted(((d, t) for d in range(len(boxes)) for t in range(len(preds)) if aff[d, t] > 0),
                               key=lambda p: -aff[p[0], p[1]]):
                if d in assigned_d or t in assigned_t:
                    continue
                self.tracks[t].update(boxes[d], scores[d], classes[d], self.frame_idx)
                assigned_d.add(d)
                assigned_t.add(t)
        for t_i, t in enumerate(self.tracks):
            if t_i not in assigned_t:
                t.misses += 1
        for d in range(len(boxes)):
            if d not in assigned_d:
                self.tracks.append(Track(boxes[d], scores[d], classes[d], self.frame_idx))
                self.stats["created"] += 1
        alive = []
        for t in self.tracks:
            if t.misses > self.max_misses:
                if t.hits < self.confirm_hits:
                    self.stats["expired_unconfirmed"] += 1
                continue
            if t.hits == self.confirm_hits and t.misses == 0:
                self.stats["confirmed"] += 1
            alive.append(t)
        self.tracks = alive
        return self.active()

    def active(self):
        return [t for t in self.tracks if t.misses == 0]

    def confirmed(self):
        return [t for t in self.tracks if t.hits >= self.confirm_hits and t.misses == 0]

    def noise_like(self):
        return [t for t in self.active() if t.hits < self.confirm_hits and self.frame_idx - t.first >= self.confirm_hits]
