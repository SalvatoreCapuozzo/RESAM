"""Overlay drawing (OpenCV, BGR) shared by the dashboard, evaluation and export tools."""

import cv2
import numpy as np

# BGR, chosen to stay readable on both the black/orange and green/yellow radar palettes
CLASS_COLORS = {
    1: (255, 200, 0),    # Ship / Target       - cyan-blue
    2: (180, 180, 180),  # Noise / Clutter     - grey
    3: (255, 0, 255),    # Unknown TGT / Land* - magenta
    4: (0, 140, 255),    # Coast/Port          - orange
    5: (0, 255, 0),      # My Ship             - green
    6: (0, 0, 255),      # AIS Lost            - red
}


def color(cid):
    return CLASS_COLORS.get(int(cid), (255, 255, 255))


def overlay_semantic(frame, sem, alpha=0.45):
    if sem is None or not sem.any():
        return frame.copy()  # always a new image: callers draw on the result
    col = np.zeros_like(frame)
    for c in np.unique(sem):
        if c:
            col[sem == c] = color(c)
    m = sem > 0
    out = frame.copy()
    out[m] = (frame[m] * (1 - alpha) + col[m] * alpha).astype(np.uint8)
    return out


def draw_boxes(frame, boxes, classes, scores=None, names=None, thickness=2, labels=True):
    for i, (b, c) in enumerate(zip(boxes, classes)):
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        col = color(c)
        cv2.rectangle(frame, (x0, y0), (x1, y1), col, thickness)
        if labels:
            txt = (names or {}).get(int(c), str(int(c)))
            if scores is not None:
                txt += f" {scores[i]:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x0, y0 - th - 4), (x0 + tw + 2, y0), col, -1)
            cv2.putText(frame, txt, (x0 + 1, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_tracks(frame, tracks, confirm_hits=3):
    for t in tracks:
        pts = np.array(t.history, np.int32)
        col = color(t.cls) if t.hits >= confirm_hits else (120, 120, 120)
        if len(pts) > 1:
            cv2.polylines(frame, [pts], False, col, 1, cv2.LINE_AA)
        x0, y0 = int(t.box[0]), int(t.box[3]) + 11
        cv2.putText(frame, f"#{t.id}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    return frame


def draw_roi(frame, roi, col=(90, 90, 90)):
    cv2.circle(frame, (int(roi.cx), int(roi.cy)), int(roi.r), col, 1, cv2.LINE_AA)
    return frame


def render(frame, result, names=None, roi=None, tracks=None, show_semantic=True, gt=None):
    out = overlay_semantic(frame, result.semantic) if show_semantic else frame.copy()
    if roi is not None:
        draw_roi(out, roi)
    if gt is not None:
        for b in gt[0]:
            x0, y0, x1, y1 = [int(round(v)) for v in b]
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 1, cv2.LINE_4)
    draw_boxes(out, result.boxes, result.classes, result.scores, names)
    if tracks:
        draw_tracks(out, tracks)
    return out
