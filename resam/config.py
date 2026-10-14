"""Class definitions, logical class mappings and radar display profiles."""

import numpy as np

# Annotation classes as written by radar_labeler.py (categories.txt, 1-based).
CLASSES = {1: "Ship", 2: "Noise", 3: "Unknown TGT", 4: "Coast/Port", 5: "My Ship", 6: "AIS Lost"}
VALID_CLASS_IDS = set(CLASSES)

# Logical groupings. Measured on the labels (tools/audit_dataset.py): 70% of "Ship" boxes
# contain the display's AIS triangle glyph (42% glyph only, no echo), while "Unknown TGT" and
# "AIS Lost" are 88-96% plain echoes that differ only by AIS history. `visual` is therefore
# the grouping a pixel-only model can learn honestly; the AIS split is recovered downstream
# by geo-fusing detections with the AIS NMEA logs (radar_labeler already decodes them).
MAPPINGS = {
    "original": {
        "desc": "Original classes",
        "lut": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
        "classes": dict(CLASSES),
    },
    "mapping_1": {
        "desc": "Unknown & AIS Lost -> Ship",
        "lut": {1: 1, 2: 2, 3: 1, 4: 4, 5: 5, 6: 1},
        "classes": {1: "Ship (merged)", 2: "Noise", 4: "Coast/Port", 5: "My Ship"},
    },
    "mapping_2": {
        "desc": "AIS Lost -> Ship",
        "lut": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 1},
        "classes": {1: "Ship (merged)", 2: "Noise", 3: "Unknown TGT", 4: "Coast/Port", 5: "My Ship"},
    },
    "visual": {
        "desc": "Appearance-learnable classes (Target / Clutter / Land / Own ship)",
        "lut": {1: 1, 2: 2, 3: 1, 4: 3, 5: 4, 6: 1},
        "classes": {1: "Target", 2: "Clutter", 3: "Land", 4: "Own Ship"},
    },
}


def remap_array(arr, mapping_key):
    """Apply a mapping LUT to a label array (0 = background stays 0, unknown ids -> 0)."""
    lut = np.zeros(256, dtype=np.uint8)
    for src, dst in MAPPINGS[mapping_key]["lut"].items():
        lut[src] = dst
    return lut[np.clip(arr, 0, 255).astype(np.uint8)]


def remap_id(cid, mapping_key):
    return MAPPINGS[mapping_key]["lut"].get(int(cid))


# Radar display profiles, measured from the dataset (static-edge circle fit over many
# frames, see resam.geometry.fit_display_circle). `circle` = (cx, cy, r) of the radar
# video ring in full-resolution pixels.
DISPLAY_PROFILES = {
    "jrc_orange_1280x1024": {"size": (1280, 1024), "circle": (510, 514, 479)},
    "green_1024x768": {"size": (1024, 768), "circle": (404, 384, 359)},
}


def profile_for_size(width, height):
    for name, p in DISPLAY_PROFILES.items():
        if p["size"] == (width, height):
            return name, p
    return None, None
