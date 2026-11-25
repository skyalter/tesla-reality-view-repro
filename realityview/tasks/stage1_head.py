"""Stage 1, second part: train the Gaussian head on the frozen backbone.

    model   frozen backbone + depth head (keeps training) + appearance branch + exposure correction
    data    one moment t of one log (7 cameras x the input moments) and the 7 photos of one other moment s
    step    one Gaussian per input pixel -> render the 7 cameras at t and the 7 cameras at s -> compare

Which pixels produce a Gaussian: everything outside the ego body; in the views of the earlier input moments
also not the pixels on moving objects and persons (they would be drawn where they no longer are).

Where a render is compared with its photo:
    cameras at t    everywhere outside the ego body
    cameras at s    outside the ego body, outside the boxes of moving objects and persons at s, not where
                    Gaussians of objects that moved since t are drawn, and only where the inputs have seen
                    the scene at all (coverage)

Losses: per-pixel depth against LiDAR and sky pushed far away (on the input views, as in the backbone
training); photo mean squared error after exposure correction; LPIPS, faded in; rendered depth against LiDAR.
"""
from __future__ import annotations

import hashlib
import os

import torch
from torch.utils.data import Subset

from ..engine.config import merge
from ..losses import depth as D
from ..losses import render as R
from .stage1_backbone import Retry, read_logs

DEFAULTS = {
    "model": {
        "name": "da3-large-1.1",        # or da3-base (for debugging on a small GPU)
        "weights_root": None,           # directory holding <name>/model.safetensors
        "backbone_checkpoint": None,    # last.pt of the backbone training; None = the original DA3 weights
        "pose_scale": 20.0,             # must be the value the backbone was trained with
        "kappa": 1.0,                   # global depth factor when there is no backbone checkpoint
        "train_depth_branch": True,     # the depth head keeps training
        "exposure": True,
        "head_checkpoint": False,       # trade time for memory in the head
        "scale_range": [0.1, 8.0],      # size of a Gaussian in pixel footprints
        "max_scale_rel": 0.02,          # and at most this fraction of its distance to the camera
        "lr_depth_head": 2.0e-4,
        "lr_appearance": 2.0e-4,
        "lr_exposure": 2.0e-4,
        "exposure_delay_steps": 1000,   # the exposure correction stays at the identity this long
    },
    "data": {
        "root": None,                   # processed root (.../av2_v1)
        "train_split": "train",
        "train_logs": "data_lists/av2_train_stage1.txt",
        "train_first": None,
        "eval_split": "val",
        "eval_logs": "data_lists/av2_val_dev10.txt",
        "eval_first": None,
        "eval_per_log": 4,              # fixed moments per log
        "eval_offsets": [0, 10, 20],    # targets of an evaluation moment: t itself, t + 0.5 s, t + 1 s
        "input_offsets": [0, -10],      # must contain 0
        "front_center_mode": "crop",
        "depth_layer": "acc",
        "sky": True,
        "min_displacement_m": 0.5,      # training skips moments in which the car stands still
        "max_target_frames": 20,        # a random target is within +-1 s of t ...
        "min_gap_frames": 3,            # ... and more than 0.15 s away from every input moment
        "retries": 3,
    },
    "render": {
        "mode": "antialiased",          # gsplat rasterize mode: antialiased | classic
        "chunk": 7,                     # cameras per rasterization call
        "coverage_min": 0.5,            # a pixel of a camera at s counts as seen by the inputs above this
        "coverage_footprint": 1.0,
        "mover_max": 0.1,               # ... and is skipped when this much of it comes from moved objects
    },
    "loss": {
        "lidar": {"weight": 1.0, "range": [0.5, 200.0]},
        "sky": {"weight": 0.1, "distance": 400.0, "low": 0.3, "high": 0.5, "window": 5},
        "photo": {"weight": 1.0},
        "lpips": {"weight": 0.05, "ramp_steps": 1000, "net": "vgg",
                  "max_views": None},     # LPIPS on at most this many of the rendered cameras per sample
                                          # (a random choice each step); None = all. Saves memory.
        "render_depth": {"weight": 0.5, "alpha_min": 0.5},
    },
    "eval": {"panels": 6},              # evaluation samples whose photo / render comparison is saved as an image
}


def load_trained_model(checkpoint: str, weights_root: str | None = None, data_root: str | None = None):
    """Rebuild the model of a stage1_head run from its checkpoint (``last.pt``).

    Returns (model in evaluation mode on the CPU, task). The checkpoint holds every weight, so no backbone
    checkpoint is needed; ``weights_root`` (the directory with <name>/model.safetensors, used only to build
    the network) and ``data_root`` replace the paths stored in the checkpoint's configuration, which are
    those of the machine the run was made on.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(state["cfg"])
    cfg["model"] = {**(cfg.get("model") or {}), "backbone_checkpoint": None}
    if weights_root:
        cfg["model"]["weights_root"] = weights_root
    if data_root:
        cfg["data"] = {**(cfg.get("data") or {}), "root": data_root}
    task = Stage1HeadTask(cfg)
    model = task.build_model()
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not fit the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
    task.step = int(state.get("step", 0))
    return model.eval(), task


def offset_name(k: int) -> str:
    """Name of a target offset in metric keys: t, p10 (10 frames after t), m4 (4 frames before t)."""
    return "t" if k == 0 else f"{'p' if k > 0 else 'm'}{abs(k)}"


class Stage1HeadTask:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.m = merge(DEFAULTS["model"], cfg.get("model") or {})
        self.d = merge(DEFAULTS["data"], cfg.get("data") or {})
        self.r = merge(DEFAULTS["render"], cfg.get("render") or {})
        self.l = merge(DEFAULTS["loss"], cfg.get("loss") or {})
        self.e = merge(DEFAULTS["eval"], cfg.get("eval") or {})
        for section, key in ((self.m, "weights_root"), (self.d, "root")):
            if not section[key]:
                raise ValueError(f"{'model' if section is self.m else 'data'}.{key} is not set")
        offsets = [int(o) for o in self.d["input_offsets"]]
        if 0 not in offsets:
            raise ValueError("data.input_offsets must contain 0: the cameras at t are rendered and compared")
        self.t_slot = offsets.index(0)
        self.seed = int(cfg.get("seed", 0))
        self.run_dir = cfg.get("run_dir")
        self.lpips = None
        self.backbone_info = None
        self.step = 0                   # steps finished so far (for the names of saved evaluation images)

    # ------------------------------------------------------------------ model
    def build_model(self):
        from ..models.gs_head import Stage1HeadModel

        m = self.m
        weights = os.path.join(m["weights_root"], m["name"], "model.safetensors")
        model = Stage1HeadModel(m["name"], weights, backbone_checkpoint=m["backbone_checkpoint"] or None,
                                pose_scale=m["pose_scale"], kappa=m["kappa"],
                                train_depth_branch=m["train_depth_branch"], exposure=m["exposure"],
                                head_checkpoint=m["head_checkpoint"], scale_range=m["scale_range"],
                                max_scale_rel=m["max_scale_rel"])
        self.backbone_info = model.backbone_checkpoint
        return model

    def on_device(self, device) -> None:
        if self.l["lpips"]["weight"] > 0:
            self.lpips = R.PerceptualLoss(self.l["lpips"]["net"]).to(device)

    def param_groups(self, model):
        parts = model.param_groups()
        m = self.m
        groups = [{"name": "depth_head", "params": parts["depth_head"], "lr": m["lr_depth_head"]},
                  {"name": "appearance", "params": parts["appearance"], "lr": m["lr_appearance"]}]
        if "exposure" in parts:
            groups.append({"name": "exposure", "params": parts["exposure"], "lr": m["lr_exposure"],
                           "delay_steps": int(m["exposure_delay_steps"])})
        return [g for g in groups if g["params"]]

    # ------------------------------------------------------------------ data
    def _dataset(self, split, logs, evaluation: bool):
        from ..data.head_dataset import Stage1HeadDataset

        d = self.d
        return Stage1HeadDataset(d["root"], split, logs, input_offsets=tuple(d["input_offsets"]),
                                 front_center_mode=d["front_center_mode"], depth_layer=d["depth_layer"],
                                 with_sky=d["sky"],
                                 target_offsets=list(d["eval_offsets"]) if evaluation else "random",
                                 min_displacement_m=0.0 if evaluation else d["min_displacement_m"],
                                 max_target_frames=d["max_target_frames"], min_gap_frames=d["min_gap_frames"],
                                 seed=self.seed)

    def train_logs(self) -> list:
        return read_logs(self.d["train_logs"], self.d["train_first"])

    def eval_logs(self) -> list:
        return read_logs(self.d["eval_logs"], self.d["eval_first"]) if self.d["eval_logs"] else []

    def build_train_dataset(self):
        d = self.d
        train, evaluation = self.train_logs(), self.eval_logs()
        if d["train_split"] != d["eval_split"] and set(train) & set(evaluation):
            raise RuntimeError("the same log is in the training and the evaluation list")
        dataset = self._dataset(d["train_split"], train, False)
        if dataset.skipped:
            raise RuntimeError(f"training logs that are not ready: {dataset.skipped}")
        return Retry(dataset, d["retries"])

    def build_eval_dataset(self):
        from ..data.splits import val_fixed_moments

        d = self.d
        if not d["eval_logs"]:
            return None
        dataset = self._dataset(d["eval_split"], self.eval_logs(), True)
        if dataset.skipped:
            raise RuntimeError(f"evaluation logs that are not ready: {dataset.skipped}")
        by_log = {}
        for index, (log_index, moment) in enumerate(dataset.samples):
            by_log.setdefault(log_index, {}).setdefault(moment, []).append(index)
        chosen = []
        for log_index, moments in sorted(by_log.items()):
            n = dataset.n_moments[log_index]
            wanted = [n // 2] if int(d["eval_per_log"]) == 1 else val_fixed_moments(n, d["eval_per_log"])
            picked = sorted({min(moments, key=lambda moment: abs(moment - t)) for t in wanted})
            for moment in picked:           # every target offset of the chosen moments
                chosen.extend(moments[moment])
        return Subset(dataset, chosen)

    def checkpoint_extra(self) -> dict:
        def digest(ids):
            return hashlib.sha256("\n".join(ids).encode()).hexdigest()

        train, evaluation = self.train_logs(), self.eval_logs()
        return {"data_lists": {"train_split": self.d["train_split"], "train_logs": len(train),
                               "train_sha256": digest(train), "eval_split": self.d["eval_split"],
                               "eval_logs": len(evaluation), "eval_sha256": digest(evaluation)},
                "backbone_checkpoint": self.backbone_info}

    # ------------------------------------------------------------------ what is rendered and compared
    def gaussian_masks(self, batch):
        """(keep, mover), both (B, N, H, W): the pixels that produce a Gaussian, and the ones on moving
        objects or persons (flagged, so that where they are drawn at another moment can be skipped)."""
        from ..data.boxes import MASK_MOVING, MASK_PERSON

        moving = (batch["box_mask"] & (MASK_MOVING | MASK_PERSON)) > 0
        earlier = (batch["slot"] != self.t_slot)[:, :, None, None]
        keep = ~batch["ego_mask"] & ~(earlier & moving)
        return keep, moving & keep

    def targets(self, batch, b: int, with_t: bool = True) -> dict:
        """The cameras sample b is rendered into: the M cameras at t (``with_t``) and the M cameras at s."""
        sel = batch["slot"][b] == self.t_slot
        at_s = {"images": batch["tgt_images"][b], "K": batch["tgt_K"][b], "ref_T_cam": batch["tgt_ref_T_cam"][b],
                "ego": batch["tgt_ego_mask"][b], "box": batch["tgt_box_mask"][b], "depth": batch["tgt_depth"][b],
                "flag": batch["tgt_depth_flag"][b], "cam_id": batch["tgt_cam_id"][b]}
        M = at_s["images"].shape[0]
        novel = torch.ones(M, dtype=torch.bool, device=sel.device)
        if int(batch["tgt_offset"][b]) == 0:        # the target moment is t itself: nothing has moved
            novel = ~novel
        if not with_t:
            return {**at_s, "novel": novel, "exposure_index": torch.arange(M, device=sel.device)}
        at_t = {"images": batch["images"][b][sel], "K": batch["K"][b][sel], "ref_T_cam": batch["ref_T_cam"][b][sel],
                "ego": batch["ego_mask"][b][sel], "box": batch["box_mask"][b][sel], "depth": batch["depth"][b][sel],
                "flag": batch["depth_flag"][b][sel], "cam_id": batch["cam_id"][b][sel]}
        if not torch.equal(at_t["cam_id"], at_s["cam_id"]):
            raise RuntimeError("the cameras at t and at s are not in the same order")
        out = {k: torch.cat([at_t[k], at_s[k]]) for k in at_t}
        out["novel"] = torch.cat([torch.zeros_like(novel), novel])
        out["exposure_index"] = torch.arange(M, device=sel.device).repeat(2)
        return out

    def render_targets(self, out: dict, batch: dict, b: int, tgt: dict) -> dict:
        """Render sample b into its target cameras and decide where the result is compared with the photos."""
        from ..data.boxes import MASK_MOVING, MASK_PERSON
        from ..models.gs_head import ExposureCorrection
        from ..render.rasterize import coverage, render

        r = self.r
        g = out["gaussians"][b]
        H, W = tgt["images"].shape[-2:]
        res = render(g, tgt["ref_T_cam"], tgt["K"], H, W, background=out["background"], mode=r["mode"],
                     chunk=r["chunk"])
        compare = ~tgt["ego"]
        seen = torch.ones_like(res["alpha"])
        novel = tgt["novel"]
        if bool(novel.any()):
            seen[novel] = coverage(g, tgt["ref_T_cam"][novel], tgt["K"][novel], H, W,
                                   footprint=r["coverage_footprint"], chunk=r["chunk"])
            moved_now = (tgt["box"] & (MASK_MOVING | MASK_PERSON)) > 0
            drawn_stale = res["mover"].detach() > r["mover_max"]
            compare = compare & ~(novel[:, None, None] & (moved_now | drawn_stale | (seen <= r["coverage_min"])))
        rgb = res["rgb"]
        if out["exposure"] is not None:
            A, offset = out["exposure"]
            sel = batch["slot"][b] == self.t_slot
            index = tgt["exposure_index"]
            rgb = ExposureCorrection.apply(rgb, A[b][sel][index], offset[b][sel][index])
        photo = tgt["images"].float() / 255.0 if tgt["images"].dtype == torch.uint8 else tgt["images"].float()
        return {"rgb": rgb, "photo": photo, "compare": compare, "depth": res["depth"], "alpha": res["alpha"],
                "coverage": seen}

    # ------------------------------------------------------------------ loss
    def _classes(self, batch):
        sky = self.l["sky"]
        return D.pixel_classes(batch["depth"], batch["depth_flag"], batch["ego_mask"], batch.get("sky"),
                               lidar_range=tuple(self.l["lidar"]["range"]), sky_low=sky["low"],
                               sky_high=sky["high"], sky_window=sky["window"])

    def lpips_weight(self, step: int) -> float:
        cfg = self.l["lpips"]
        ramp = max(1, int(cfg["ramp_steps"]))
        return cfg["weight"] * min(1.0, (step + 1) / ramp)

    def losses(self, out: dict, batch: dict, step: int):
        """(total, {term: value}, {name: number for the log})."""
        l = self.l
        log_z = out["log_depth"]
        classes = self._classes(batch)
        terms = {"lidar": D.lidar_depth_loss(log_z, batch["depth"], classes["lidar"])}
        total = l["lidar"]["weight"] * terms["lidar"]
        if "sky" in batch and l["sky"]["weight"] > 0:
            norms = D.ray_norms(batch["K"], *log_z.shape[-2:])
            terms["sky"] = D.sky_distance_loss(log_z, norms, classes["sky"], l["sky"]["distance"])
            total = total + l["sky"]["weight"] * terms["sky"]

        B = log_z.shape[0]
        w_lpips = self.lpips_weight(step) if self.lpips is not None else 0.0
        photo = lpips = rdepth = 0.0
        share = alpha = 0.0
        for b in range(B):
            tgt = self.targets(batch, b)
            res = self.render_targets(out, batch, b, tgt)
            photo = photo + R.masked_mse(res["rgb"], res["photo"], res["compare"]) / B
            if w_lpips > 0:
                views = torch.arange(res["rgb"].shape[0], device=res["rgb"].device)
                limit = l["lpips"]["max_views"]
                if limit and len(views) > int(limit):
                    views = views[torch.randperm(len(views), device=views.device)[:int(limit)]]
                lpips = lpips + self.lpips(res["rgb"][views], res["photo"][views], res["compare"][views]) / B
            if l["render_depth"]["weight"] > 0:
                mask = R.rendered_depth_mask(res["depth"], res["alpha"], tgt["depth"], tgt["flag"], res["compare"],
                                             alpha_min=l["render_depth"]["alpha_min"],
                                             lidar_range=tuple(l["lidar"]["range"]))
                rdepth = rdepth + R.rendered_depth_loss(res["depth"], tgt["depth"], mask) / B
            share += float(res["compare"].float().mean()) / B
            alpha += float(res["alpha"].detach()[res["compare"]].mean()) / B if bool(res["compare"].any()) else 0.0
        terms["photo"] = photo
        total = total + l["photo"]["weight"] * photo
        if w_lpips > 0:
            terms["lpips"] = lpips
            total = total + w_lpips * lpips
        if l["render_depth"]["weight"] > 0:
            terms["render_depth"] = rdepth
            total = total + l["render_depth"]["weight"] * rdepth
        info = {"psnr": R.psnr(float(photo.detach())), "share_compared": share, "alpha": alpha,
                "gaussians_m": sum(len(g) for g in out["gaussians"]) / B / 1e6, "lpips_weight": w_lpips}
        return total, terms, info

    def training_step(self, model, batch, step):
        keep, mover = self.gaussian_masks(batch)
        out = model(batch["images"], batch["ref_T_cam"], batch["K"], batch["dt"], keep, mover)
        total, terms, info = self.losses(out, batch, step)
        logs = {f"loss_{name}": float(value.detach()) for name, value in terms.items()}
        logs.update(info)
        self.step = step + 1
        return total, logs

    # ------------------------------------------------------------------ evaluation
    def summary(self, m: dict) -> str:
        parts = []
        for k in self.d["eval_offsets"]:
            name = offset_name(int(k))
            if f"psnr_{name}" in m:
                label = "t" if k == 0 else f"t{k / 20:+.2f}s"
                parts.append(f"{label}: PSNR {m[f'psnr_{name}']:.2f} LPIPS {m.get(f'lpips_{name}', float('nan')):.3f} "
                             f"compared {100 * m[f'compared_{name}']:.0f}% "
                             f"depth {m.get(f'render_depth_absrel_{name}', float('nan')):.3f}")
        parts.append(f"pixel depth AbsRel {m.get('absrel_pixel', float('nan')):.4f}")
        return "  |  ".join(parts) + f"  [{m.get('samples', 0):.0f} samples]"

    def evaluate(self, model, loader, device) -> dict:
        """Renders against photos on the fixed evaluation moments, per target offset.

        psnr / lpips: inside the compared region, after exposure correction (psnr_raw: before it);
        compared: share of the pixels outside the ego body that are compared; render_depth_absrel: rendered
        depth against LiDAR; alpha: mean rendered opacity in the compared region. absrel_pixel is the
        per-pixel depth of the input views against LiDAR (the depth branch must not get worse)."""
        from ..engine.trainer import to_device

        sums, counts = {}, {}

        def add(name, value, weight=1.0):
            sums[name] = sums.get(name, 0.0) + float(value) * weight
            counts[name] = counts.get(name, 0.0) + weight

        lidar_range = tuple(self.l["lidar"]["range"])
        saved = 0
        for batch in loader:
            batch = to_device(batch, device)
            keep, mover = self.gaussian_masks(batch)
            out = model(batch["images"], batch["ref_T_cam"], batch["K"], batch["dt"], keep, mover)
            classes = self._classes(batch)
            lidar = classes["lidar"]
            if bool(lidar.any()):
                rel = (out["depth"] - batch["depth"]).abs() / batch["depth"].clamp(min=1e-3)
                add("absrel_pixel", rel[lidar].mean())
            for b in range(out["depth"].shape[0]):
                name = offset_name(int(batch["tgt_offset"][b]))
                tgt = self.targets(batch, b, with_t=False)
                res = self.render_targets(out, batch, b, tgt)
                compare = res["compare"]
                n = float(compare.sum())
                if n > 0:
                    add(f"mse_{name}", R.masked_mse(res["rgb"], res["photo"], compare), n)
                    raw = self.render_targets({**out, "exposure": None}, batch, b, tgt)["rgb"]
                    add(f"mse_raw_{name}", R.masked_mse(raw, res["photo"], compare), n)
                    add(f"alpha_{name}", res["alpha"][compare].mean())
                if self.lpips is not None:
                    add(f"lpips_{name}", self.lpips(res["rgb"], res["photo"], compare))
                add(f"compared_{name}", compare.float().sum() / (~tgt["ego"]).float().sum().clamp(min=1.0))
                mask = R.rendered_depth_mask(res["depth"], res["alpha"], tgt["depth"], tgt["flag"], compare,
                                             alpha_min=self.l["render_depth"]["alpha_min"], lidar_range=lidar_range)
                if bool(mask.any()):
                    rel = (res["depth"] - tgt["depth"]).abs() / tgt["depth"].clamp(min=1e-3)
                    add(f"render_depth_absrel_{name}", rel[mask].mean())
                add("samples", 1.0, 1.0)
                if saved < int(self.e["panels"]) and self.run_dir:
                    self.save_panel(res, batch, b, name, saved)
                    saved += 1
        result = {}
        for key, total in sums.items():
            if key == "samples":
                result[key] = total
            elif key.startswith("mse_raw_"):
                result["psnr_raw_" + key[len("mse_raw_"):]] = R.psnr(total / counts[key])
            elif key.startswith("mse_"):
                result["psnr_" + key[len("mse_"):]] = R.psnr(total / counts[key])
            else:
                result[key] = total / counts[key]
        return result

    def save_panel(self, res: dict, batch: dict, b: int, name: str, index: int) -> None:
        """One image per evaluation sample: the photos (top), the renders (middle) and the renders with the
        pixels that are not compared darkened (bottom), one column per camera."""
        import cv2
        import numpy as np

        photo, rgb = res["photo"], res["rgb"].clamp(0, 1)
        dimmed = rgb * (0.25 + 0.75 * res["compare"][:, None].float())
        rows = [torch.cat(list(x), dim=-1) for x in (photo, rgb, dimmed)]         # each (3, H, M * W)
        image = (torch.cat(rows, dim=-2).permute(1, 2, 0) * 255.0).round().byte().cpu().numpy()
        log_id = batch["log_id"][b] if isinstance(batch["log_id"], (list, tuple)) else str(batch["log_id"])
        folder = os.path.join(self.run_dir, "eval_images", f"step_{self.step:06d}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{index:02d}_{log_id[:8]}_{int(batch['moment'][b]):03d}_{name}.jpg")
        cv2.imwrite(path, np.ascontiguousarray(image[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 90])

