"""Training-time reader for stage 1 (Gaussian head): the stage-1 inputs plus one target moment.

One sample = (moment t of one log, target offset k). It holds everything ``Stage1Dataset`` returns (with
``box_mask`` always and ``sky`` when ``with_sky``), plus the M views of moment t + k (M = views per moment,
7, or 8 for front-center mode "split") that the rendered Gaussians are compared with:

    ref_T_ego        float32 (4, 4) ego frame (rear-axle centre, x forward, y left, z up) at the timestamp of
                     the front-center image at t -> reference (the front-center camera at t)
    city_T_ref       float64 (4, 4) reference -> city; relates the reference frames of different moments
    tgt_images       uint8 (M, 3, H, W); order of ``CAMS``, as one slot of the inputs
    tgt_K, tgt_ref_T_cam, tgt_dt, tgt_cam_id, tgt_view, tgt_timestamp_ns
                     as the input keys, first dimension M; same reference frame, dt relative to t
    tgt_depth, tgt_depth_flag   float32 / uint8 (M, H, W), layer ``depth_layer``
    tgt_ego_mask     bool (M, H, W)
    tgt_box_mask     uint8 (M, H, W) 1 moving, 2 person-like, 4 any box (boxes of the target moment)
    tgt_sky          float32 (M, H, W) (``with_sky``)
    tgt_offset       int64 () k, front-center frames (20 Hz) relative to t

Moments: ``valid_stage2`` and valid for ``input_offsets`` (``valid_moments``); with min_displacement_m > 0
also ``disp_win_max_m >= min_displacement_m`` (the car stands still otherwise).

Targets:
- ``target_offsets="random"`` (training): one sample per moment. Candidates are the integers k in
  [-max_target_frames, max_target_frames] with |k - o| > min_gap_frames for every input offset o and moment
  t + k complete; the pick is sha256(f"{seed}:{log_id}:{i}") (first 8 bytes, big-endian) modulo the number
  of candidates, so an index always gives the same sample. Moments without a candidate are dropped.
- ``target_offsets=[k1, k2, ...]`` (evaluation): one sample per (moment, k), moment-major, for every k with
  t + k present and complete; k may be 0 or an input offset (the target is then that input moment).

``self.samples[idx]`` stays (log index, moment); ``self.sample_offsets[idx]`` is k.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from .config import FRONT_CENTER
from .dataset import Stage1Dataset, valid_moments
from .geometry import se3_inv


def target_candidates(complete: np.ndarray, i: int, input_offsets, max_target_frames: int,
                      min_gap_frames: int) -> list[int]:
    """Offsets k allowed as random target of moment i (see the module docstring), ascending."""
    n = len(complete)
    out = []
    for k in range(-int(max_target_frames), int(max_target_frames) + 1):
        if any(abs(k - o) <= min_gap_frames for o in input_offsets):
            continue
        j = i + k
        if 0 <= j < n and complete[j]:
            out.append(k)
    return out


def pick_target(candidates, seed: int, log_id: str, i: int) -> int:
    h = hashlib.sha256(f"{seed}:{log_id}:{i}".encode()).digest()
    return candidates[int.from_bytes(h[:8], "big") % len(candidates)]


class Stage1HeadDataset(Stage1Dataset):
    def __init__(self, processed_root: str, split: str, log_ids, *, input_offsets=(0, -10),
                 front_center_mode: str = "crop", depth_layer: str = "acc", load_single_layer: bool = False,
                 with_sky: bool = True, target_offsets="random", min_displacement_m: float = 0.5,
                 max_target_frames: int = 20, min_gap_frames: int = 3, seed: int = 0,
                 require_checks: bool = True, allow_missing_logs: bool = False):
        super().__init__(processed_root, split, log_ids, input_offsets=input_offsets,
                         front_center_mode=front_center_mode, depth_layer=depth_layer,
                         load_single_layer=load_single_layer, with_box_masks=True, with_sky=with_sky,
                         require_checks=require_checks, allow_missing_logs=allow_missing_logs)
        if isinstance(target_offsets, str):
            if target_offsets != "random":
                raise ValueError("target_offsets must be 'random' or a list of frame offsets")
            self.target_offsets = "random"
        else:
            self.target_offsets = tuple(int(k) for k in target_offsets)
            if not self.target_offsets:
                raise ValueError("target_offsets is empty")
        self.min_displacement_m = float(min_displacement_m)
        self.max_target_frames = int(max_target_frames)
        self.min_gap_frames = int(min_gap_frames)
        self.seed = int(seed)
        # rebuild the sample table with the stage-2 rules
        self.samples = []
        self.sample_offsets = []
        for li, (log_id, store) in enumerate(self.logs):
            moments = store.parquet("moments.parquet")
            if "valid_stage2" not in moments:
                raise KeyError(f"{log_id}: moments.parquet has no valid_stage2 column")
            complete = moments["complete"].to_numpy(bool)
            ok = np.zeros(len(moments), bool)
            ok[valid_moments(moments, self.offsets)] = True
            ok &= moments["valid_stage2"].to_numpy(bool)
            if self.min_displacement_m > 0:
                ok &= moments["disp_win_max_m"].to_numpy(np.float64) >= self.min_displacement_m
            for i in np.flatnonzero(ok).tolist():
                if self.target_offsets == "random":
                    cand = target_candidates(complete, i, self.offsets, self.max_target_frames,
                                             self.min_gap_frames)
                    if cand:
                        self.samples.append((li, i))
                        self.sample_offsets.append(pick_target(cand, self.seed, log_id, i))
                else:
                    for k in self.target_offsets:
                        if 0 <= i + k < len(complete) and complete[i + k]:
                            self.samples.append((li, i))
                            self.sample_offsets.append(k)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        li, i = self.samples[idx]
        k = self.sample_offsets[idx]
        L = self._log(li)
        cam_T_ref, t_ref = self._reference(li, i)
        frame = int(L["moments"].iloc[i][f"idx_{FRONT_CENTER}"])
        ref_T_ego = cam_T_ref @ L["city_T_ego"][L["key"][(FRONT_CENTER, frame)]]
        out["ref_T_ego"] = torch.from_numpy(ref_T_ego.astype(np.float32))
        out["city_T_ref"] = torch.from_numpy(se3_inv(cam_T_ref))
        out.update(self._moment_tensors(self._read_moment(li, i + k, cam_T_ref), t_ref, prefix="tgt_"))
        out["tgt_offset"] = torch.tensor(k, dtype=torch.int64)
        return out
