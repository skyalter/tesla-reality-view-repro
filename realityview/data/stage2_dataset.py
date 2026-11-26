"""Training-time reader for stage 2 (the diffusion model).

One sample = a clip of ``clip_frames`` consecutive moments (default 41 = 2 s at 20 Hz) seen from one target
camera, plus the inputs of M conditioning moments spread over the clip (every ``cond_every_frames`` moments,
default 8, so M = 6). Each conditioning moment has the usual 14 input views of stage 1 (7 cameras at the
moment and 10 frames earlier).

Two kinds of target:
    chase   the chase-camera images rendered by ``scripts/render_targets.py`` with their confidence;
    real    the photos of one of the 7 real cameras (``scripts/make_video_frames.py``). That camera is then
            removed from the inputs (``view_valid`` is False for both of its views): otherwise the model
            would only have to copy it.

The sample index decides everything (clip, kind, camera), as the training loop requires: with both kinds
available an index runs over (clip start) x (7 chase + 7 real cameras), so the two kinds are half and half.

Keys of a sample (M moments, N = 14 views, T clip frames, F = 1 + (T - 1) / 4 latent frames, (h, w) the
latent grid = target size / 8):
    images        uint8 (M, N, 3, 392, 518); the removed camera's images are zero
    K             float32 (M, N, 3, 3)
    ref_T_cam     float32 (M, N, 4, 4)  camera -> reference of its own moment (front-center camera at the
                  moment): what the backbone is used to
    dt            float32 (M, N)  image time - moment time, seconds
    cam_id, slot  int64 (M, N)
    view_valid    bool (N,)
    clip_T_cam    float32 (M, N, 4, 4)  camera -> clip reference (front-center camera at the first clip frame)
    src_time      float32 (M, N)  image time - time of the first clip frame, seconds
    target        uint8 (T, 3, Ht, Wt) RGB
    tgt_K         float32 (3, 3) at (Ht, Wt), integer pixel centres
    tgt_ego_T_cam float32 (4, 4)  target camera -> ego frame (rear axle), constant over the clip
    tgt_clip_T_cam float32 (F, 4, 4)  target camera -> clip reference at the last frame of every latent frame
    tgt_time      float32 (F,)  time of those frames - time of the first clip frame, seconds
    conf          float32 (T, h, w)  confidence of the target in 0..1 (1 for photos)
    supervise     float32 (h, w)  1 - share of the ego vehicle in the latent position
    kind          int64 ()  0 chase, 1 real;   tgt_cam  int64 () camera index or -1
    log_id, moment (first clip frame)
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch

from .config import CAMS, FRONT_CENTER
from .dataset import Stage1Dataset
from .geometry import se3_inv
from .images import crop_intrinsics, scale_intrinsics

KIND_CHASE, KIND_REAL = 0, 1


def fit_size(src_wh, out_wh):
    """How an image of ``src_wh`` is brought to ``out_wh`` without distortion: scale so that it covers the
    output, then cut the centre. Returns (scaled (w, h), (col0, row0))."""
    s = max(out_wh[0] / src_wh[0], out_wh[1] / src_wh[1])
    w, h = max(out_wh[0], int(round(src_wh[0] * s))), max(out_wh[1], int(round(src_wh[1] * s)))
    return (w, h), ((w - out_wh[0]) // 2, (h - out_wh[1]) // 2)


def fit_image(img: np.ndarray, out_wh, nearest: bool = False) -> np.ndarray:
    src_wh = (img.shape[1], img.shape[0])
    if src_wh == tuple(out_wh):
        return img
    (w, h), (c0, r0) = fit_size(src_wh, out_wh)
    inter = cv2.INTER_NEAREST if nearest else (cv2.INTER_AREA if w < src_wh[0] else cv2.INTER_LINEAR)
    return cv2.resize(img, (w, h), interpolation=inter)[r0:r0 + out_wh[1], c0:c0 + out_wh[0]]


def fit_intrinsics(K: np.ndarray, src_wh, out_wh) -> np.ndarray:
    if tuple(src_wh) == tuple(out_wh):
        return np.array(K, np.float64)
    (w, h), (c0, r0) = fit_size(src_wh, out_wh)
    return crop_intrinsics(scale_intrinsics(K, src_wh, (w, h)), r0, c0)


def soft_confidence(alpha: np.ndarray, visibility: np.ndarray, depth_consistency: np.ndarray,
                    alpha_min: float = 0.5) -> np.ndarray:
    """The same formula as ``render.targets.soft_confidence`` (kept here so that data-loading processes do
    not import the renderer): alpha * (0.5 + 0.5 * min(1, 2 V)) * sqrt(D), 0 where alpha < alpha_min."""
    conf = alpha * (0.5 + 0.5 * np.minimum(2.0 * visibility, 1.0)) * np.sqrt(np.maximum(depth_consistency, 0.0))
    return np.where(alpha >= alpha_min, conf, 0.0).astype(np.float32)


def pool_to(x: np.ndarray, wh) -> np.ndarray:
    return cv2.resize(x.astype(np.float32), tuple(wh), interpolation=cv2.INTER_AREA)


class Stage2Dataset(Stage1Dataset):
    """
    Args (besides those of ``Stage1Dataset``):
        chase_root: output directory of ``scripts/render_targets.py`` (holding <split>/<log id>/), or None
        video_root: output directory of ``scripts/make_video_frames.py``, or None
        clip_frames: frames per clip, 1 + 4 k
        cond_every_frames: frames between conditioning moments (4 x the model's ``cond_every``)
        size: (width, height) of the target, multiples of 16
        start_stride: use every n-th possible clip start
        kinds: "both" | "chase" | "real"
        alpha_min: confidence is 0 where the rendered opacity is below this
    """

    def __init__(self, processed_root: str, split: str, log_ids, *, chase_root=None, video_root=None,
                 clip_frames: int = 41, cond_every_frames: int = 8, size=(832, 480), start_stride: int = 1,
                 kinds: str = "both", alpha_min: float = 0.5, input_offsets=(0, -10), front_center_mode: str = "crop",
                 require_checks: bool = True, allow_missing_logs: bool = False):
        super().__init__(processed_root, split, log_ids, input_offsets=input_offsets,
                         front_center_mode=front_center_mode, require_checks=require_checks,
                         allow_missing_logs=allow_missing_logs)
        if (clip_frames - 1) % 4 or cond_every_frames % 4:
            raise ValueError("clip_frames must be 1 + 4 k and cond_every_frames a multiple of 4")
        if size[0] % 16 or size[1] % 16:
            raise ValueError("the target size must be a multiple of 16")
        if len(self.layout) != len(CAMS):
            raise ValueError("stage 2 needs one view per camera (front_center_mode 'crop' or 'letterbox')")
        if kinds not in ("both", "chase", "real"):
            raise ValueError("kinds must be both, chase or real")
        self.chase_root = chase_root if kinds != "real" else None
        self.video_root = video_root if kinds != "chase" else None
        if self.chase_root is None and self.video_root is None:
            raise ValueError("neither chase targets nor video frames are given")
        self.T = int(clip_frames)
        self.F = (self.T - 1) // 4 + 1
        self.cond_step = int(cond_every_frames)
        self.moment_offsets = list(range(0, self.T, self.cond_step))
        self.size = (int(size[0]), int(size[1]))
        self.latent_wh = (self.size[0] // 8, self.size[1] // 8)
        self.alpha_min = float(alpha_min)
        self.variants = ([-1] * len(CAMS) if self.chase_root else []) + (list(range(len(CAMS))) if self.video_root else [])
        if self.chase_root and not self.video_root:
            self.variants = [-1]
        self.chase = {}         # log index -> {"folder", "camera", "ts": {moment: timestamp}}
        self.video = {}         # log index -> {"folder", "cameras"}
        self.starts = []        # (log index, first moment)
        for li, (log_id, store) in enumerate(self.logs):
            complete = store.parquet("moments.parquet")["complete"].to_numpy(bool)
            if self.chase_root:
                folder = os.path.join(self.chase_root, split, log_id)
                if not os.path.isfile(os.path.join(folder, "frames.tsv")):
                    self.skipped[log_id] = "no chase targets"
                    continue
                with open(os.path.join(folder, "camera.json")) as f:
                    camera = json.load(f)
                rows = np.loadtxt(os.path.join(folder, "frames.tsv"), delimiter="\t", skiprows=1, usecols=(0, 1),
                                  dtype=np.int64, ndmin=2)
                self.chase[li] = {"folder": folder, "camera": camera, "ts": {int(m): int(t) for m, t in rows}}
            if self.video_root:
                folder = os.path.join(self.video_root, split, log_id)
                if not os.path.isfile(os.path.join(folder, "DONE")):
                    self.skipped[log_id] = "no video frames (run scripts/make_video_frames.py)"
                    self.chase.pop(li, None)
                    continue
                with open(os.path.join(folder, "cameras.json")) as f:
                    self.video[li] = {"folder": folder, "cameras": json.load(f)}
            lo = -min(min(self.offsets), 0)
            found = []
            for i0 in range(lo, len(complete) - self.T + 1):
                if not complete[i0 + min(self.offsets):i0 + self.T].all():
                    continue
                if li in self.chase and any(i not in self.chase[li]["ts"] for i in range(i0, i0 + self.T)):
                    continue
                found.append(i0)
            self.starts.extend((li, i0) for i0 in found[::max(1, int(start_stride))])

    def __len__(self):
        return len(self.starts) * len(self.variants)

    # ------------------------------------------------------------------ targets
    def _read_rgb(self, path: str) -> np.ndarray:
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise IOError(f"cannot read {path}")
        return im

    def _chase_target(self, li, L, frames):
        info = self.chase[li]
        cam = info["camera"]
        src_wh = (int(cam["width"]), int(cam["height"]))
        K = fit_intrinsics(np.array(cam["K"], np.float64), src_wh, self.size)
        ego_T_cam = np.array(cam["ego_T_cam"], np.float64)
        rgb = np.zeros((self.T, self.size[1], self.size[0], 3), np.uint8)
        conf = np.zeros((self.T, self.latent_wh[1], self.latent_wh[0]), np.float32)
        city_T_cam, ts = [], []
        for n, i in enumerate(frames):
            t = info["ts"][i]
            rgb[n] = fit_image(self._read_rgb(os.path.join(info["folder"], "rgb", f"{t}.jpg")), self.size)[:, :, ::-1]
            parts = []
            for sub in ("alpha", "vis", "dcons"):
                x = cv2.imread(os.path.join(info["folder"], sub, f"{t}.png"), cv2.IMREAD_GRAYSCALE)
                if x is None:
                    raise IOError(f"cannot read {info['folder']}/{sub}/{t}.png")
                parts.append(x.astype(np.float32) / 255.0)
            conf[n] = pool_to(fit_image(soft_confidence(*parts, alpha_min=self.alpha_min), self.size), self.latent_wh)
            frame = int(L["moments"].iloc[i][f"idx_{FRONT_CENTER}"])
            city_T_cam.append(L["city_T_ego"][L["key"][(FRONT_CENTER, frame)]] @ ego_T_cam)
            ts.append(int(L["moments"].iloc[i][f"ts_{FRONT_CENTER}"]))
        ego = cv2.imread(os.path.join(info["folder"], "ego_mask.png"), cv2.IMREAD_GRAYSCALE)
        if ego is None:
            raise IOError(f"cannot read {info['folder']}/ego_mask.png")
        ego = fit_image((ego > 0).astype(np.uint8), self.size, nearest=True)
        return rgb, conf, ego, K, ego_T_cam, np.stack(city_T_cam), np.array(ts, np.int64)

    def _real_target(self, li, L, frames, cam: str):
        info = self.video[li]
        meta = info["cameras"][cam]
        src_wh = (int(meta["width"]), int(meta["height"]))
        K = fit_intrinsics(np.array(meta["K"], np.float64), src_wh, self.size)
        rgb = np.zeros((self.T, self.size[1], self.size[0], 3), np.uint8)
        city_T_cam, ts = [], []
        for n, i in enumerate(frames):
            row = L["moments"].iloc[i]
            t = int(row[f"ts_{cam}"])
            rgb[n] = fit_image(self._read_rgb(os.path.join(info["folder"], cam, f"{t}.jpg")), self.size)[:, :, ::-1]
            city_T_cam.append(self._city_T_cam(L, cam, int(row[f"idx_{cam}"])))
            ts.append(t)
        ego = cv2.imread(os.path.join(info["folder"], cam, "ego_mask.png"), cv2.IMREAD_GRAYSCALE)
        if ego is None:
            raise IOError(f"cannot read {info['folder']}/{cam}/ego_mask.png")
        ego = fit_image((ego > 0).astype(np.uint8), self.size, nearest=True)
        conf = np.ones((self.T, self.latent_wh[1], self.latent_wh[0]), np.float32)
        return rgb, conf, ego, K, L["ego_T_cam"][cam], np.stack(city_T_cam), np.array(ts, np.int64)

    # ------------------------------------------------------------------ sample
    def __getitem__(self, idx):
        li, i0 = self.starts[idx // len(self.variants)]
        variant = self.variants[idx % len(self.variants)]
        log_id, _ = self.logs[li]
        L = self._log(li)
        cam_T_clip, t0 = self._reference(li, i0)            # clip reference = front-center camera at i0
        n_views = len(self.layout) * len(self.offsets)
        per = {k: [] for k in ("images", "K", "ref_T_cam", "dt", "cam_id", "clip_T_cam", "src_time")}
        for off in self.moment_offsets:
            i = i0 + off
            cam_T_ref, t_ref = self._reference(li, i)
            parts = [self._read_moment(li, i + o, cam_T_clip) for o in self.offsets]
            m = {key: np.concatenate([p[key] for p in parts]) for key in ("images", "K", "ref_T_cam", "timestamp_ns", "cam_id")}
            clip_T_cam = m["ref_T_cam"]                                     # float64, relative to the clip reference
            per["images"].append(m["images"])
            per["K"].append(m["K"])
            per["clip_T_cam"].append(clip_T_cam)
            per["ref_T_cam"].append((cam_T_ref @ se3_inv(cam_T_clip)) @ clip_T_cam)
            per["dt"].append((m["timestamp_ns"] - t_ref) / 1e9)
            per["src_time"].append((m["timestamp_ns"] - t0) / 1e9)
            per["cam_id"].append(m["cam_id"])
        images = np.stack(per["images"]).transpose(0, 1, 4, 2, 3)           # (M, N, 3, H, W)
        cam_id = np.stack(per["cam_id"])
        slot = np.broadcast_to(np.repeat(np.arange(len(self.offsets), dtype=np.int64), len(self.layout)),
                               cam_id.shape).copy()
        frames = list(range(i0, i0 + self.T))
        if variant < 0:
            rgb, conf, ego, K, ego_T_cam, city_T_cam, ts = self._chase_target(li, L, frames)
            view_valid = np.ones(n_views, bool)
        else:
            rgb, conf, ego, K, ego_T_cam, city_T_cam, ts = self._real_target(li, L, frames, CAMS[variant])
            view_valid = cam_id[0] != variant
            images = images * view_valid[None, :, None, None, None]
        latent_frames = np.arange(self.F) * 4                               # last frame of every latent frame
        return {
            "images": torch.from_numpy(np.ascontiguousarray(images)),
            "K": torch.from_numpy(np.stack(per["K"]).astype(np.float32)),
            "ref_T_cam": torch.from_numpy(np.stack(per["ref_T_cam"]).astype(np.float32)),
            "dt": torch.from_numpy(np.stack(per["dt"]).astype(np.float32)),
            "cam_id": torch.from_numpy(cam_id),
            "slot": torch.from_numpy(slot),
            "view_valid": torch.from_numpy(view_valid),
            "clip_T_cam": torch.from_numpy(np.stack(per["clip_T_cam"]).astype(np.float32)),
            "src_time": torch.from_numpy(np.stack(per["src_time"]).astype(np.float32)),
            "target": torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))),
            "tgt_K": torch.from_numpy(K.astype(np.float32)),
            "tgt_ego_T_cam": torch.from_numpy(np.asarray(ego_T_cam).astype(np.float32)),
            "tgt_clip_T_cam": torch.from_numpy((cam_T_clip @ city_T_cam[latent_frames]).astype(np.float32)),
            "tgt_time": torch.from_numpy(((ts[latent_frames] - t0) / 1e9).astype(np.float32)),
            "conf": torch.from_numpy(conf),
            "supervise": torch.from_numpy(1.0 - pool_to(ego, self.latent_wh)),
            "kind": torch.tensor(KIND_CHASE if variant < 0 else KIND_REAL, dtype=torch.int64),
            "tgt_cam": torch.tensor(variant, dtype=torch.int64),
            "log_id": log_id,
            "moment": i0,
        }
