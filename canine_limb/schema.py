"""Original AP-10K anatomy; left/right are the animal's own sides."""

LIMBS = {
    "left-fore": ((5, 6, 7), ("shoulder", "elbow", "front_paw")),
    "right-fore": ((8, 9, 10), ("shoulder", "elbow", "front_paw")),
    "left-hind": ((11, 12, 13), ("hip", "knee", "hind_paw")),
    "right-hind": ((14, 15, 16), ("hip", "knee", "hind_paw")),
}

COLORS = {
    "left-fore": (80, 240, 80),
    "right-fore": (0, 200, 255),
    "left-hind": (255, 180, 80),
    "right-hind": (230, 100, 240),
}

CSV_FIELDS = [
    "frame", "time_s", "limb", "joint", "x_px", "y_px", "score", "status",
    "dog_score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
]


def selected_limbs(limb):
    return tuple(LIMBS) if limb == "all" else (limb,)
