"""Colour-based radar echo extraction and a classical (training-free) blob detector.

Measured on the dataset (OpenCV HSV, H in 0..180): echoes are orange/yellow
(H ~ 5..42, high S); range rings, heading line, VRM/EBL and UI glyphs are green/white/blue.
Grayscale conversion (as done by the training notebook) destroys exactly this cue.
"""

import cv2
import numpy as np

ECHO_HSV_LO = (5, 90, 70)
ECHO_HSV_HI = (42, 255, 255)


def echo_mask(image_bgr, lo=ECHO_HSV_LO, hi=ECHO_HSV_HI):
    """uint8 {0,255} mask of radar-echo-coloured pixels."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


def echo_channel(image_bgr):
    """Single-channel 'echo strength' image (V where hue is echo-like, else 0). Useful as an
    extra input channel or as a colour-invariant replacement for the grayscale conversion."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(ECHO_HSV_LO, np.uint8), np.array(ECHO_HSV_HI, np.uint8))
    return cv2.bitwise_and(hsv[..., 2], m)


class ClassicalDetector:
    """Echo blob detector, comparable to what a radar ARPA front-end does before tracking.

    Classes (original ids): large blobs -> Coast/Port (4); blob containing the own-ship
    position -> My Ship (5); everything else -> Ship (1). Noise (2) is decided by the tracker
    (blobs that do not persist across sweeps), see resam.tracking.
    """

    def __init__(self, land_area_frac=0.004, min_area=6, close_px=3, own_ship_xy=None):
        self.land_area_frac = land_area_frac
        self.min_area = min_area
        self.close_px = close_px
        self.own_ship_xy = own_ship_xy

    def __call__(self, image_bgr, roi_mask=None):
        m = echo_mask(image_bgr)
        if roi_mask is not None:
            m = cv2.bitwise_and(m, m, mask=roi_mask)
        if self.close_px:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_px, self.close_px))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        land_area = self.land_area_frac * m.shape[0] * m.shape[1]
        boxes, scores, classes = [], [], []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            cls = 4 if area >= land_area else 1
            if self.own_ship_xy is not None:
                ox, oy = self.own_ship_xy
                if x <= ox <= x + w and y <= oy <= y + h and area < land_area:
                    cls = 5
            boxes.append([x, y, x + w, y + h])
            # pseudo-confidence: fill ratio * size prior (bigger and denser = more certain)
            scores.append(float(min(1.0, (area / max(w * h, 1)) * min(1.0, area / 60.0) + 0.2)))
            classes.append(cls)
        return (np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32),
                np.asarray(classes, np.int32), lab, m)
