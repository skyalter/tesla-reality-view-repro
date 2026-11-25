"""Processing parameters of the offline Argoverse 2 preprocessing (version ``av2_v1``).

Every value that changes the processed files lives in ``DEFAULT_CONFIG``. The full dict is written to
``CONFIG.yaml`` in the output root and its sha256 into every log's ``meta.json``. Changing any value
means a new output root (``av2_v2``, ...); an existing root with a different config is never overwritten.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

# Camera order used everywhere (frames tables, samples). The front-center camera comes first because it
# defines the moments and the reference frame of a sample.
CAMS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
)
FRONT_CENTER = "ring_front_center"

# Categories that are treated as dynamic regardless of their estimated speed ("person-like").
PERSON_LIKE = (
    "PEDESTRIAN",
    "BICYCLIST",
    "MOTORCYCLIST",
    "WHEELED_RIDER",
    "OFFICIAL_SIGNALER",
    "STROLLER",
    "WHEELCHAIR",
    "WHEELED_DEVICE",
    "DOG",
    "ANIMAL",
)

DEFAULT_CONFIG = {
    "version": "av2_v1",
    "images": {
        # stored sizes (width, height); landscape cameras and the portrait front-center camera
        "landscape_wh": [518, 392],
        "front_center_wh": [518, 684],
        "jpeg_quality": 95,
        "interpolation": "INTER_AREA",
        # rows [146, 538) of the stored 518x684 front-center image give the 518x392 landscape crop
        # (full-resolution rows 437..1610)
        "front_center_crop_row0": 146,
    },
    "index": {
        "past_offset_frames": 10,  # t - 0.5 s at 20 Hz
        "group_tol_ms": 30.0,  # a moment is complete if every camera has a frame within this of t
        "stage2_margin_frames": 20,  # valid_stage2 needs 20 <= i <= n - 21
        "disp_window_frames": 20,  # +-1 s window for the maximum ego displacement
    },
    "lidar": {
        "sweeps_each_side": 5,  # accumulated layer: sweeps k0-5 .. k0+5
        "sync_sweeps_each_side": 1,  # synchronised moving-object points from k0-1 .. k0+1
        "sync_tol_ms": 25.0,  # |sweep_ts + offset_ns - camera_ts| for points inside dynamic boxes
        "z_min_m": 0.5,
        "z_max_m": 250.0,
        "lidar_box_margin_lw_m": 0.3,  # dynamic boxes grow by this in length and width
        "lidar_box_margin_h_m": 0.2,  # and by this in height, when removing points from the static set
        "seethrough_window": 5,
        "seethrough_ratio": 1.2,  # flag 4 if depth > ratio * min depth in the window
    },
    "boxes": {
        "speed_half_window_s": 0.5,  # speed from centres ~1 s apart (+-0.5 s, clipped to the track)
        "speed_min_span_s": 0.25,  # shorter spans give no speed -> treated as moving
        "moving_speed_mps": 0.5,
        "max_extrapolation_s": 0.05,
        "max_gap_s": 0.15,  # do not interpolate across a gap in the track longer than this
    },
    "masks": {
        # applied when box masks are drawn at read time (not stored)
        "margin_lw_m": 0.3,
        "margin_h_m": 0.2,
        "velocity_stretch_s": 0.1,
        "dilate_px": 2,
        "face_samples": 9,
    },
    "ego_mask": {
        "dilate_px": 1,
    },
}

FLAG_CENTER_SWEEP = 1
FLAG_SYNC_DYNAMIC = 2
FLAG_SEETHROUGH = 4


def default_config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def config_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def load_root_config(root: str) -> dict:
    import yaml

    with open(os.path.join(root, "CONFIG.yaml")) as f:
        return yaml.safe_load(f)


def init_root_config(root: str, cfg: dict) -> str:
    """Write CONFIG.yaml / CONFIG.sha256 into a new output root, or verify an existing one matches.

    Returns the sha256. Raises if the root already holds a different config (use a new version root).
    """
    import yaml

    os.makedirs(root, exist_ok=True)
    sha = config_sha256(cfg)
    f_sha = os.path.join(root, "CONFIG.sha256")
    if os.path.isfile(f_sha):
        with open(f_sha) as f:
            old = f.read().strip()
        if old != sha:
            raise ValueError(f"{root} was produced with a different config (sha256 {old[:12]} != {sha[:12]}); "
                             "write to a new output root instead")
        return sha
    with open(os.path.join(root, "CONFIG.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    with open(f_sha, "w") as f:
        f.write(sha + "\n")
    return sha
