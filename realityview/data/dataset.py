"""Training-time reader for stage 1 (backbone): 7 cameras x input moments, first version.

One sample = one moment t of one log. For every input offset (default t and t - 10 frames = t - 0.5 s)
and every camera (order of ``CAMS``; the front-center camera gives one or two views, see below) it
returns, slot-major:

    images      uint8  (N, 3, H, W) RGB, H x W = 392 x 518
    K           float32 (N, 3, 3) at 518x392, integer pixel-centre convention (a renderer with pixel centres
                at j + 0.5, such as gsplat, needs cx, cy + 0.5: ``images.intrinsics_for_half_pixel_renderer``)
    ref_T_cam   float32 (N, 4, 4) camera -> reference; reference = camera frame of the front-center camera at
                t (its own timestamp), also when t itself is not among the inputs; metres (no scale
                applied); computed in float64
    dt          float32 (N,) image timestamp - front-center timestamp at t, seconds
    cam_id      int64 (N,) index into CAMS;  slot int64 (N,) index into input_offsets
    view        int64 (N,) 0, or 1 for the second front-center view of mode "split"
    timestamp_ns int64 (N,)
    depth       float32 (N, H, W) LiDAR depth of the chosen layer (metres along the optical axis), 0 = none
    depth_flag  uint8 (N, H, W) 1 centre sweep, 2 synchronised dynamic point, 4 suspected see-through
    ego_mask    bool (N, H, W) ego body (and, for "letterbox", the padded columns): never supervised
    displacement_m  float32 () largest ego displacement within +-1 s of t (below 0.5 m the car is standing)
  optional:
    depth_single, depth_single_flag   the single-sweep layer (``load_single_layer=True``)
    box_mask    uint8 (N, H, W) 1 moving, 2 person-like, 4 any box (``with_box_masks=True``)
    sky         float32 (N, H, W) raw sky score of DA3METRIC-LARGE (``with_sky=True``): 0 on ground and
                buildings, around 1 on sky; thresholds are applied by the loss. Needs scripts/sky_scores.py.

Normalisation is left to the GPU.

Two switches are configuration:
- ``input_offsets``: which moments are inputs, in front-center frames relative to t (default (0, -10));
- ``front_center_mode``: how the stored 518x684 front-center image becomes 518x392 input
  ("crop" default: rows 146..538; "split": two views, rows 0..392 and 292..684; "letterbox": whole image
  shrunk to 297x392 and padded), see ``images.front_center_views``.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CAMS, FRONT_CENTER, load_root_config
from .geometry import se3_from_quat_trans, se3_inv
from .images import FRONT_CENTER_MODES, front_center_views, stored_size
from .splits import check_root_splits
from .store import SKY_DONE, LogStore, find_log, img_relpath

SKY_UINT8_SCALE = 100.0     # stored sky value = round(raw score * 100); the same constant as in models/sky.py


def valid_moments(moments, offsets) -> np.ndarray:
    """Moments i for which every i + offset exists and is complete (all 7 cameras within tolerance).

    The reference moment t = i itself must be complete too (it defines the reference frame and dt). With the
    default offsets (0, -10) this is exactly the ``valid_stage1`` column of moments.parquet."""
    complete = moments["complete"].to_numpy(bool)
    n = len(complete)
    ok = complete.copy()
    i = np.arange(n)
    for o in offsets:
        j = i + o
        inside = (j >= 0) & (j < n)
        ok &= inside
        ok[inside] &= complete[j[inside]]
    return i[ok]


def load_root_ego_masks(processed_root: str, cfg: dict) -> dict[str, np.ndarray]:
    """The stored-resolution ego masks of a processed root, one per camera (written by preprocess.py)."""
    out = {}
    for cam in CAMS:
        p = os.path.join(processed_root, "ego_masks", f"{cam}.png")
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.isfile(p) else None
        if m is None:
            raise FileNotFoundError(f"ego mask missing: {p} (written by scripts/preprocess.py)")
        w, h = stored_size(cam, cfg)
        if m.shape != (h, w):
            raise ValueError(f"ego mask {p} has shape {m.shape}, expected {(h, w)}")
        out[cam] = m > 0
    return out


class Stage1Dataset(Dataset):
    def __init__(self, processed_root: str, split: str, log_ids, input_offsets=(0, -10),
                 front_center_mode: str = "crop", depth_layer: str = "acc", load_single_layer: bool = False,
                 with_box_masks: bool = False, with_sky: bool = False, require_checks: bool = True,
                 allow_missing_logs: bool = False):
        if front_center_mode not in FRONT_CENTER_MODES:
            raise ValueError(f"front_center_mode must be one of {FRONT_CENTER_MODES}")
        if depth_layer not in ("acc", "single"):
            raise ValueError("depth_layer must be 'acc' or 'single'")
        self.root = processed_root
        self.split = split
        self.offsets = tuple(int(o) for o in input_offsets)
        if not self.offsets:
            raise ValueError("input_offsets is empty")
        self.fc_mode = front_center_mode
        self.depth_layer = depth_layer
        self.load_single = load_single_layer
        self.with_box_masks = with_box_masks
        self.with_sky = with_sky
        self.cfg = load_root_config(processed_root)
        self.split_lists = check_root_splits(processed_root)  # raises if train / dev / val overlap
        self.fc_views = front_center_views(front_center_mode, self.cfg)
        self.ego = load_root_ego_masks(processed_root, self.cfg)
        # (camera, view) per slot, in output order
        self.layout = [(c, v) for c in CAMS for v in (range(len(self.fc_views)) if c == FRONT_CENTER else [0])]
        self.logs = []  # (log_id, store)
        self.n_moments = []  # per log: number of moments (front-center frames)
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
            if with_sky and not store.exists(SKY_DONE):
                self.skipped[log_id] = "no sky scores (run scripts/sky_scores.py)"
                continue
            moments = store.parquet("moments.parquet")
            li = len(self.logs)
            self.logs.append((log_id, store))
            self.n_moments.append(len(moments))
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
            boxes, box_rows = None, None
            if self.with_box_masks:
                boxes = store.parquet("boxes.parquet")
                box_rows = {(c, int(f)): idx for (c, f), idx in boxes.groupby(["cam", "frame"]).indices.items()}
            self._cache[li] = {"meta": meta, "frames": frames, "moments": moments, "city_T_ego": T, "key": key,
                               "boxes": boxes, "box_rows": box_rows,
                               "ego_T_cam": {c: np.array(meta["ego_T_cam"][c], np.float64) for c in CAMS},
                               "K": {c: np.array(meta["K_stored"][c], np.float64) for c in CAMS}}
        return self._cache[li]

    def _city_T_cam(self, L, cam, frame):
        return L["city_T_ego"][L["key"][(cam, frame)]] @ L["ego_T_cam"][cam]

    def _reference(self, li, i):
        """(cam_T_ref, timestamp_ns) of the reference: the front-center camera at moment i, float64."""
        L = self._log(li)
        row = L["moments"].iloc[i]
        frame = int(row[f"idx_{FRONT_CENTER}"])
        return se3_inv(self._city_T_cam(L, FRONT_CENTER, frame)), int(row[f"ts_{FRONT_CENTER}"])

    def _read_moment(self, li, j, cam_T_ref):
        """Every view of moment j of log li, in ``self.layout`` order, as numpy arrays.

        Keys: images (M, H, W, 3) RGB, K, ref_T_cam (float64, reference given by cam_T_ref), timestamp_ns,
        cam_id, view, depth, depth_flag, ego_mask, and when switched on depth_single, depth_single_flag,
        box_mask, sky (uint8, raw score x 100)."""
        from .boxes import draw_box_mask

        log_id, store = self.logs[li]
        L = self._log(li)
        row = L["moments"].iloc[j]
        W, H = (int(x) for x in self.cfg["images"]["landscape_wh"])
        n = len(self.layout)
        images = np.zeros((n, H, W, 3), np.uint8)
        Ks = np.zeros((n, 3, 3))
        city_T_cam = np.zeros((n, 4, 4))
        ts_all = np.zeros(n, np.int64)
        cam_id = np.zeros(n, np.int64)
        view_id = np.zeros(n, np.int64)
        depth = np.zeros((n, H, W), np.float32)
        dflag = np.zeros((n, H, W), np.uint8)
        ego = np.zeros((n, H, W), bool)
        layers = [("depth", self.depth_layer)] + ([("single", "single")] if self.load_single else [])
        if self.load_single:
            depth_s = np.zeros((n, H, W), np.float32)
            dflag_s = np.zeros((n, H, W), np.uint8)
        if self.with_box_masks:
            box_mask = np.zeros((n, H, W), np.uint8)
        if self.with_sky:
            sky = np.zeros((n, H, W), np.uint8)
        k = 0
        for cam in CAMS:
            frame = int(row[f"idx_{cam}"])
            ts = int(row[f"ts_{cam}"])
            im = cv2.imdecode(np.frombuffer(store.read(img_relpath(cam, ts)), np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                raise IOError(f"cannot decode {log_id}/{img_relpath(cam, ts)}")
            hw = im.shape[:2]
            K = L["K"][cam]
            cTw = self._city_T_cam(L, cam, frame)  # city_T_cam
            sparse = {name: store.depth(layer, cam, ts) for name, layer in layers}
            bm = None
            if self.with_box_masks:
                rows = L["box_rows"].get((cam, frame))
                bx = L["boxes"].iloc[rows] if rows is not None else L["boxes"].iloc[0:0]
                bm = draw_box_mask(bx, se3_inv(cTw), K, (hw[1], hw[0]), self.cfg)
            sk = store.sky(cam, ts) if self.with_sky else None
            if sk is not None and sk.shape != hw:
                raise ValueError(f"{log_id}: sky score of {cam} {ts} has shape {sk.shape}, image {hw}")
            views = self.fc_views if cam == FRONT_CENTER else [None]
            for v_i, view in enumerate(views):
                if view is None:
                    images[k], Ks[k], ego[k] = im[:, :, ::-1], K, self.ego[cam]
                    d = {name: sp.densify(hw) for name, sp in sparse.items()}
                    if bm is not None:
                        box_mask[k] = bm
                    if sk is not None:
                        sky[k] = sk
                else:
                    images[k] = view.image(im)[:, :, ::-1]
                    Ks[k] = view.K(K, hw[1])
                    ego[k] = view.mask(self.ego[cam], pad_value=True)
                    d = {}
                    for name, sp in sparse.items():
                        vu, z, f = view.sparse(sp.vu, sp.z, sp.flag, hw[1])
                        dd = np.zeros((H, W), np.float32)
                        ff = np.zeros((H, W), np.uint8)
                        dd[vu[:, 0], vu[:, 1]] = z.astype(np.float32)
                        ff[vu[:, 0], vu[:, 1]] = f
                        d[name] = (dd, ff)
                    if bm is not None:
                        box_mask[k] = view.mask(bm)
                    if sk is not None:
                        sky[k] = view.image(sk)  # padded columns get 0 = not sky (they are ego-masked anyway)
                depth[k], dflag[k] = d["depth"]
                if self.load_single:
                    depth_s[k], dflag_s[k] = d["single"]
                city_T_cam[k] = cTw
                ts_all[k] = ts
                cam_id[k] = CAMS.index(cam)
                view_id[k] = v_i
                k += 1
        out = {"images": images, "K": Ks, "ref_T_cam": cam_T_ref @ city_T_cam, "timestamp_ns": ts_all,
               "cam_id": cam_id, "view": view_id, "depth": depth, "depth_flag": dflag, "ego_mask": ego}
        if self.load_single:
            out["depth_single"], out["depth_single_flag"] = depth_s, dflag_s
        if self.with_box_masks:
            out["box_mask"] = box_mask
        if self.with_sky:
            out["sky"] = sky
        return out

    @staticmethod
    def _moment_tensors(m, t_ref, prefix=""):
        """Torch output keys (with ``prefix``) from ``_read_moment`` arrays; dt relative to t_ref."""
        out = {
            "images": torch.from_numpy(np.ascontiguousarray(m["images"].transpose(0, 3, 1, 2))),
            "K": torch.from_numpy(m["K"].astype(np.float32)),
            "ref_T_cam": torch.from_numpy(m["ref_T_cam"].astype(np.float32)),
            "dt": torch.from_numpy(((m["timestamp_ns"] - t_ref) / 1e9).astype(np.float32)),
            "cam_id": torch.from_numpy(m["cam_id"]),
            "view": torch.from_numpy(m["view"]),
            "timestamp_ns": torch.from_numpy(m["timestamp_ns"]),
            "depth": torch.from_numpy(m["depth"]),
            "depth_flag": torch.from_numpy(m["depth_flag"]),
            "ego_mask": torch.from_numpy(m["ego_mask"]),
        }
        for key in ("depth_single", "depth_single_flag", "box_mask"):
            if key in m:
                out[key] = torch.from_numpy(m[key])
        if "sky" in m:
            out["sky"] = torch.from_numpy(m["sky"].astype(np.float32) / SKY_UINT8_SCALE)
        return {prefix + key: v for key, v in out.items()}

    def __getitem__(self, idx):
        li, i = self.samples[idx]
        log_id, _ = self.logs[li]
        cam_T_ref, t_ref = self._reference(li, i)
        parts = [self._read_moment(li, i + o, cam_T_ref) for o in self.offsets]
        m = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        per = self._moment_tensors(m, t_ref)
        slot = np.repeat(np.arange(len(self.offsets), dtype=np.int64), len(self.layout))
        out = {key: per[key] for key in ("images", "K", "ref_T_cam", "dt", "cam_id")}
        out["slot"] = torch.from_numpy(slot)
        for key in ("view", "timestamp_ns", "depth", "depth_flag", "ego_mask"):
            out[key] = per[key]
        disp = float(self._log(li)["moments"].iloc[i]["disp_win_max_m"])
        out["displacement_m"] = torch.tensor(disp, dtype=torch.float32)
        out["log_id"] = log_id
        out["moment"] = i
        for key in ("depth_single", "depth_single_flag", "box_mask", "sky"):
            if key in per:
                out[key] = per[key]
        return out
