"""Training-time reader for stage 1 (backbone): 7 cameras x input moments, first version.

One sample = one moment t of one log. For every input offset (default t and t - 10 frames = t - 0.5 s)
and every camera (order of ``CAMS``) it returns, slot-major:

    images      uint8  (N, 3, H, W) RGB, H x W = 392 x 518
    K           float32 (N, 3, 3) at 518x392, integer pixel-centre convention
    ref_T_cam   float32 (N, 4, 4) camera -> reference; reference = front-center camera at t, at its own
                timestamp; metres (no scale applied); computed in float64
    dt          float32 (N,) image timestamp - front-center timestamp at t, seconds
    cam_id      int64 (N,) index into CAMS;  slot int64 (N,) index into input_offsets
    timestamp_ns int64 (N,)
    depth       float32 (N, H, W) LiDAR depth of the chosen layer, 0 = no point
    depth_flag  uint8 (N, H, W) 1 centre sweep, 2 synchronised dynamic point, 4 suspected see-through
    ego_mask    bool (N, H, W)
  optional:
    depth_single, depth_single_flag   the single-sweep layer (``load_single_layer=True``)
    box_mask    uint8 (N, H, W) 1 moving, 2 person-like, 4 any box (``with_box_masks=True``)

Sky scores are not produced yet; nothing sky-related is returned. Normalisation is left to the GPU.

Two switches are configuration:
- ``input_offsets``: which moments are inputs, in front-center frames relative to t (default (0, -10));
- ``front_center_mode``: "crop" (default, rows 146..538 of the stored 518x684 image), "split" and
  "letterbox" are reserved and raise NotImplementedError.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CAMS, FRONT_CENTER, load_root_config
from .geometry import se3_from_quat_trans, se3_inv
from .images import FRONT_CENTER_MODES, front_center_view
from .store import LogStore, find_log, img_relpath


def valid_moments(moments, offsets) -> np.ndarray:
    """Moments i for which every i + offset exists and is complete (all 7 cameras within tolerance)."""
    complete = moments["complete"].to_numpy(bool)
    n = len(complete)
    ok = np.ones(n, bool)
    i = np.arange(n)
    for o in offsets:
        j = i + o
        inside = (j >= 0) & (j < n)
        ok &= inside
        ok[inside] &= complete[j[inside]]
    return i[ok]


class Stage1Dataset(Dataset):
    def __init__(self, processed_root: str, split: str, log_ids, input_offsets=(0, -10),
                 front_center_mode: str = "crop", depth_layer: str = "acc", load_single_layer: bool = False,
                 with_box_masks: bool = False, require_checks: bool = True, allow_missing_logs: bool = False):
        if front_center_mode not in FRONT_CENTER_MODES:
            raise ValueError(f"front_center_mode must be one of {FRONT_CENTER_MODES}")
        if front_center_mode != "crop":
            raise NotImplementedError(f"front_center_mode '{front_center_mode}' is reserved but not implemented")
        if depth_layer not in ("acc", "single"):
            raise ValueError("depth_layer must be 'acc' or 'single'")
        self.root = processed_root
        self.split = split
        self.offsets = tuple(int(o) for o in input_offsets)
        self.ref_slot = self.offsets.index(0) if 0 in self.offsets else 0
        self.fc_mode = front_center_mode
        self.depth_layer = depth_layer
        self.load_single = load_single_layer
        self.with_box_masks = with_box_masks
        self.cfg = load_root_config(processed_root)
        self.ego = {}
        for cam in CAMS:
            p = os.path.join(processed_root, "ego_masks", f"{cam}.png")
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.isfile(p) else None
            self.ego[cam] = m > 0 if m is not None else None
        self.logs = []  # (log_id, store)
        self.samples = []  # (log index, moment i)
        self.skipped = {}
        for log_id in log_ids:
            path = find_log(processed_root, split, log_id)
            if path is None:
                if allow_missing_logs:
                    self.skipped[log_id] = "not processed"
                    continue
                raise FileNotFoundError(f"processed log not found: {split}/{log_id}")
            store = LogStore(path)
            if not store.exists("DONE"):
                self.skipped[log_id] = "no DONE"
                continue
            if require_checks and not store.json("DONE").get("checks_passed", False):
                self.skipped[log_id] = "checks failed"
                continue
            moments = store.parquet("moments.parquet")
            li = len(self.logs)
            self.logs.append((log_id, store))
            self.samples.extend((li, int(i)) for i in valid_moments(moments, self.offsets))
        self._cache = {}

    def __len__(self):
        return len(self.samples)

    def _log(self, li):
        if li not in self._cache:
            log_id, store = self.logs[li]
            meta = store.json("meta.json")
            frames = store.parquet("frames.parquet")
            moments = store.parquet("moments.parquet")
            T = se3_from_quat_trans(frames[["qw", "qx", "qy", "qz"]].to_numpy(np.float64),
                                    frames[["tx_m", "ty_m", "tz_m"]].to_numpy(np.float64))
            key = {(c, int(f)): n for n, (c, f) in enumerate(zip(frames["cam"], frames["frame"]))}
            boxes = store.parquet("boxes.parquet") if self.with_box_masks else None
            self._cache[li] = {"meta": meta, "frames": frames, "moments": moments, "city_T_ego": T, "key": key,
                               "boxes": boxes,
                               "ego_T_cam": {c: np.array(meta["ego_T_cam"][c]) for c in CAMS},
                               "K": {c: np.array(meta["K_stored"][c]) for c in CAMS}}
        return self._cache[li]

    def __getitem__(self, idx):
        li, i = self.samples[idx]
        log_id, store = self.logs[li]
        L = self._log(li)
        mom = L["moments"]
        H, W = self.cfg["images"]["landscape_wh"][1], self.cfg["images"]["landscape_wh"][0]
        n = len(self.offsets) * len(CAMS)
        images = np.zeros((n, H, W, 3), np.uint8)
        Ks = np.zeros((n, 3, 3))
        city_T_cam = np.zeros((n, 4, 4))
        ts_all = np.zeros(n, np.int64)
        cam_id = np.zeros(n, np.int64)
        slot = np.zeros(n, np.int64)
        depth = np.zeros((n, H, W), np.float32)
        dflag = np.zeros((n, H, W), np.uint8)
        ego = np.zeros((n, H, W), bool)
        if self.load_single:
            depth_s = np.zeros((n, H, W), np.float32)
            dflag_s = np.zeros((n, H, W), np.uint8)
        if self.with_box_masks:
            from .boxes import draw_box_mask
            box_mask = np.zeros((n, H, W), np.uint8)
        k = 0
        for s, o in enumerate(self.offsets):
            row = mom.iloc[i + o]
            for c_id, cam in enumerate(CAMS):
                frame = int(row[f"idx_{cam}"])
                ts = int(row[f"ts_{cam}"])
                im = cv2.imdecode(np.frombuffer(store.read(img_relpath(cam, ts)), np.uint8), cv2.IMREAD_COLOR)
                K = L["K"][cam]
                hw = im.shape[:2]
                d, f = store.depth(self.depth_layer, cam, ts).densify(hw)
                extra = [d, f]
                if self.load_single:
                    extra += list(store.depth("single", cam, ts).densify(hw))
                e = self.ego[cam] if self.ego[cam] is not None else np.zeros(hw, bool)
                extra.append(e)
                cTw = L["city_T_ego"][L["key"][(cam, frame)]] @ L["ego_T_cam"][cam]  # city_T_cam
                if self.with_box_masks:
                    fb = L["boxes"]
                    bx = fb[(fb["cam"] == cam) & (fb["frame"] == frame)]
                    extra.append(draw_box_mask(bx, se3_inv(cTw), K, (hw[1], hw[0]), self.cfg))
                if cam == FRONT_CENTER:
                    im, K, extra = front_center_view(im, K, self.fc_mode, self.cfg, extra)
                images[k] = im[:, :, ::-1]
                Ks[k] = K
                city_T_cam[k] = cTw
                ts_all[k] = ts
                cam_id[k] = c_id
                slot[k] = s
                depth[k], dflag[k] = extra[0], extra[1]
                j = 2
                if self.load_single:
                    depth_s[k], dflag_s[k] = extra[2], extra[3]
                    j = 4
                ego[k] = extra[j]
                if self.with_box_masks:
                    box_mask[k] = extra[j + 1]
                k += 1
        ref = self.ref_slot * len(CAMS)  # front-center camera of the reference slot
        ref_T_cam = se3_inv(city_T_cam[ref]) @ city_T_cam
        t_ref = ts_all[ref]
        out = {
            "images": torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))),
            "K": torch.from_numpy(Ks.astype(np.float32)),
            "ref_T_cam": torch.from_numpy(ref_T_cam.astype(np.float32)),
            "dt": torch.from_numpy(((ts_all - t_ref) / 1e9).astype(np.float32)),
            "cam_id": torch.from_numpy(cam_id),
            "slot": torch.from_numpy(slot),
            "timestamp_ns": torch.from_numpy(ts_all),
            "depth": torch.from_numpy(depth),
            "depth_flag": torch.from_numpy(dflag),
            "ego_mask": torch.from_numpy(ego),
            "log_id": log_id,
            "moment": i,
        }
        if self.load_single:
            out["depth_single"] = torch.from_numpy(depth_s)
            out["depth_single_flag"] = torch.from_numpy(dflag_s)
        if self.with_box_masks:
            out["box_mask"] = torch.from_numpy(box_mask)
        return out
