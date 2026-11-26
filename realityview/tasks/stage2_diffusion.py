"""Stage 2: the video diffusion model that draws the view of a target camera from the camera features.

    model   frozen stage-1 network (backbone + depth head), frozen video autoencoder, Wan2.1 diffusion
            backbone with a feature cross-attention per block, conditioner (see models/diffusion/)
    data    a clip of 41 frames from the chase camera (rendered by the Gaussian model, with confidence) or
            from one of the real cameras (photos; that camera is removed from the inputs), plus the inputs
            of 6 conditioning moments (see data/stage2_dataset.py)
    step    encode the clip to latents, add noise, predict the denoising direction, weighted squared error

What trains (``model.scope``): the new modules (conditioner, feature cross-attention) and, by default, the
self-attention of the diffusion backbone; "adapters" trains the new modules only, "full" everything.

Evaluation: on a few fixed clips, the loss at three fixed noise levels and a sampled video next to its
target, written to <run_dir>/samples/step_<n>/.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import torch
from torch.utils.data import Subset

from ..engine.config import merge
from ..losses import diffusion as D
from ..models.diffusion import schedule
from .stage1_backbone import REPO, Retry, read_logs

DEFAULTS = {
    "model": {
        "dit": "wan2.1-t2v-1.3b",       # or "tiny" (random small network for tests; needs no weights)
        "weights_root": None,           # holds wan2.1-t2v-1.3b/ and <stage1 name>/model.safetensors
        "prompt": "assets/stage2_prompt.safetensors",   # scripts/encode_prompt.py; relative to the repository
        "stage1": {
            "name": "da3-large-1.1",    # or da3-base (debugging)
            "head_checkpoint": None,    # last.pt of a stage1_head run: backbone + trained depth head (the normal case)
            "backbone_checkpoint": None,  # or last.pt of a stage1_backbone run; neither = the original DA3 weights
            "pose_scale": 20.0,
            "kappa": 1.0,               # global depth factor when there is no checkpoint
        },
        "cond_every": 2,                # one conditioning moment per this many latent frames
        "use_depth": True,              # token geometry uses the depth head; False = rays only
        "scope": "adapters+self_attn",  # adapters | adapters+self_attn | full
        "grad_checkpoint": True,
        "amp": True,                    # bf16 autocast on the GPU
        "lr_new": 1.0e-4,
        "lr_pretrained": 1.0e-5,
        "drop_prob": 0.1,               # share of samples trained without the feature tokens (for guidance)
        "shift": 5.0,                   # noise-level shift of training
        "sigma_weighting": "bell",      # bell | none
    },
    "data": {
        "root": None,                   # processed root (.../av2_v1)
        "chase_root": None,             # output of scripts/render_targets.py
        "video_root": None,             # output of scripts/make_video_frames.py
        "kinds": "both",                # both | chase | real
        "train_split": "train",
        "train_logs": "data_lists/av2_train_stage1.txt",
        "train_first": None,
        "eval_split": "val",
        "eval_logs": None,              # None = no evaluation
        "eval_first": None,
        "eval_clips": 4,                # evaluation clips in total (spread over the logs; chase and one real camera)
        "clip_frames": 41,
        "size": [832, 480],
        "start_stride": 1,
        "input_offsets": [0, -10],
        "front_center_mode": "crop",
        "retries": 3,
    },
    "loss": {"w_min": 0.05, "c_ref": 0.5, "alpha_min": 0.5},
    "sample": {"steps": 30, "guidance": 3.0, "shift": 5.0, "enabled": True},
}


def write_video(path: str, frames: np.ndarray, fps: int = 20) -> str:
    """frames (T, H, W, 3) uint8 RGB -> an .mp4 (or, without a video writer, a .jpg sheet of 6 frames)."""
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(path, list(frames), fps=fps, quality=8, macro_block_size=1)
        return path
    except Exception as error:  # noqa: BLE001 - a missing encoder must not end a run
        import cv2

        pick = np.linspace(0, len(frames) - 1, 6).round().astype(int)
        sheet = np.concatenate([frames[i] for i in pick], axis=1)
        alt = os.path.splitext(path)[0] + ".jpg"
        cv2.imwrite(alt, sheet[:, :, ::-1])
        print(f"WARNING: no video written ({error!r}); frame sheet {alt}", flush=True)
        return alt


class Stage2DiffusionTask:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.m = merge(DEFAULTS["model"], cfg.get("model") or {})
        self.d = merge(DEFAULTS["data"], cfg.get("data") or {})
        self.l = merge(DEFAULTS["loss"], cfg.get("loss") or {})
        self.s = merge(DEFAULTS["sample"], cfg.get("sample") or {})
        if not self.d["root"]:
            raise ValueError("data.root is not set")
        if self.m["dit"] != "tiny" and not self.m["weights_root"]:
            raise ValueError("model.weights_root is not set")
        self.seed = int(cfg.get("seed", 0))
        self.run_dir = cfg.get("run_dir")
        self.checkpoint_skip = ()
        self.stage1_info = None
        self.step = 0

    # ------------------------------------------------------------------ model
    def build_stage1(self):
        from ..models.gs_head import Stage1HeadModel

        s = self.m["stage1"]
        if s["head_checkpoint"]:
            from .stage1_head import load_trained_model

            model, _ = load_trained_model(s["head_checkpoint"], self.m["weights_root"], self.d["root"])
            self.stage1_info = {"head_checkpoint": str(s["head_checkpoint"])}
            return model.stage1
        weights = os.path.join(self.m["weights_root"], s["name"], "model.safetensors") if self.m["weights_root"] else None
        model = Stage1HeadModel(s["name"], weights, load_weights=weights is not None,
                                backbone_checkpoint=s["backbone_checkpoint"] or None, pose_scale=s["pose_scale"],
                                kappa=s["kappa"], exposure=False)
        self.stage1_info = {"backbone_checkpoint": model.backbone_checkpoint, "name": s["name"]}
        return model.stage1

    def build_model(self):
        from ..models.diffusion.model import Stage2Model, VideoVAE, build_dit
        from ..models.diffusion.wan_dit import CONFIGS

        m = self.m
        if m["dit"] == "tiny":
            dit = build_dit("tiny", None, m["grad_checkpoint"])
            text = torch.zeros(4, CONFIGS["tiny"]["text_dim"])
            vae = self.tiny_vae()
        else:
            from safetensors.torch import load_file

            folder = os.path.join(m["weights_root"], m["dit"])
            dit = build_dit(m["dit"], os.path.join(folder, "diffusion_pytorch_model.safetensors"), m["grad_checkpoint"])
            prompt = m["prompt"] if os.path.isabs(m["prompt"]) else os.path.join(REPO, m["prompt"])
            text = load_file(prompt)["prompt"].float()
            vae = VideoVAE(os.path.join(folder, "Wan2.1_VAE.pth"))
        model = Stage2Model(self.build_stage1(), dit, vae, text, cond_every=m["cond_every"], use_depth=m["use_depth"],
                            scope=m["scope"], amp_dtype=torch.bfloat16 if m["amp"] else None)
        self.checkpoint_skip = model.frozen_prefixes()
        return model

    @staticmethod
    def tiny_vae():
        from ..models.diffusion.toy_vae import ToyVAE

        return ToyVAE()

    def param_groups(self, model):
        parts = model.param_groups()
        groups = [{"name": "new", "params": parts["new"], "lr": self.m["lr_new"]},
                  {"name": "pretrained", "params": parts["pretrained"], "lr": self.m["lr_pretrained"]}]
        return [g for g in groups if g["params"]]

    # ------------------------------------------------------------------ data
    def _dataset(self, split, logs, **over):
        from ..data.stage2_dataset import Stage2Dataset

        d = self.d
        args = dict(chase_root=d["chase_root"], video_root=d["video_root"], clip_frames=d["clip_frames"],
                    cond_every_frames=4 * int(self.m["cond_every"]), size=tuple(d["size"]),
                    start_stride=d["start_stride"], kinds=d["kinds"], alpha_min=self.l["alpha_min"],
                    input_offsets=tuple(d["input_offsets"]), front_center_mode=d["front_center_mode"])
        args.update(over)
        return Stage2Dataset(d["root"], split, logs, **args)

    def train_logs(self) -> list:
        return read_logs(self.d["train_logs"], self.d["train_first"])

    def eval_logs(self) -> list:
        return read_logs(self.d["eval_logs"], self.d["eval_first"]) if self.d["eval_logs"] else []

    def build_train_dataset(self):
        dataset = self._dataset(self.d["train_split"], self.train_logs())
        if dataset.skipped:
            raise RuntimeError(f"training logs that are not ready: {dataset.skipped}")
        if len(dataset) == 0:
            raise RuntimeError("no training clips: are the chase targets and the video frames complete?")
        return Retry(dataset, self.d["retries"])

    def build_eval_dataset(self):
        if not self.d["eval_logs"]:
            return None
        dataset = self._dataset(self.d["eval_split"], self.eval_logs(), start_stride=1)
        if dataset.skipped:
            raise RuntimeError(f"evaluation logs that are not ready: {dataset.skipped}")
        n_var = len(dataset.variants)
        n = max(1, int(self.d["eval_clips"]))
        starts = np.linspace(0, len(dataset.starts) - 1, n + 2)[1:-1].round().astype(int)
        # alternate: the chase camera (variant 0) and a real camera (the last variant) when both exist
        chosen = [int(s) * n_var + (0 if k % 2 == 0 else n_var - 1) for k, s in enumerate(starts)]
        return Subset(dataset, chosen)

    def checkpoint_extra(self) -> dict:
        def digest(ids):
            return hashlib.sha256("\n".join(ids).encode()).hexdigest()

        train = self.train_logs()
        return {"data_lists": {"train_split": self.d["train_split"], "train_logs": len(train),
                               "train_sha256": digest(train)}, "stage1": self.stage1_info}

    # ------------------------------------------------------------------ training
    def loss_weight(self, batch) -> torch.Tensor:
        conf = D.temporal_pool(batch["conf"].float())
        supervise = batch["supervise"].float()[:, None].expand_as(conf)
        return D.confidence_weight(conf, supervise, self.l["w_min"], self.l["c_ref"])

    def training_step(self, model, batch, step):
        B = batch["images"].shape[0]
        dev = batch["images"].device
        sigma, level_weight = schedule.draw_levels(B, self.m["shift"], self.m["sigma_weighting"], device=dev)
        drop = (torch.rand(B) < self.m["drop_prob"]).to(dev)
        out = model(batch, sigma, None, drop)
        weight = self.loss_weight(batch)
        loss, per = D.weighted_flow_loss(out["pred"], out["target"], weight, level_weight)
        logs = {"mean_weight": float(weight.mean()), "sigma": float(sigma.mean())}
        for name, kind in (("chase", 0), ("real", 1)):
            pick = batch["kind"] == kind
            if bool(pick.any()):
                logs[f"loss_{name}"] = float(per[pick].mean())
        self.step = step + 1
        return loss, logs

    # ------------------------------------------------------------------ evaluation
    def summary(self, m: dict) -> str:
        return "  ".join(f"{k} {v:.4f}" for k, v in m.items() if isinstance(v, float))

    @torch.no_grad()
    def evaluate(self, model, loader, device) -> dict:
        sums, counts = {}, {}
        out_dir = os.path.join(self.run_dir, "samples", f"step_{self.step:08d}") if self.run_dir else None
        for n, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            gen = torch.Generator().manual_seed(1000 + n)
            latents = model.latents_of(batch)
            batch_l = {**batch, "latents": latents}
            weight = self.loss_weight(batch)
            noise = torch.randn(latents.shape, generator=gen).to(device)
            name = "chase" if int(batch["kind"][0]) == 0 else "real"
            for s in (0.3, 0.6, 0.9):
                sigma = torch.full((latents.shape[0],), s, device=device)
                out = model(batch_l, sigma, noise)
                _, per = D.weighted_flow_loss(out["pred"], out["target"], weight)
                for key in ("loss", f"loss_{name}"):
                    sums[key] = sums.get(key, 0.0) + float(per.mean())
                    counts[key] = counts.get(key, 0) + 1
            if out_dir and self.s["enabled"]:
                T = batch["target"].shape[1]
                hw = tuple(batch["target"].shape[-2:])
                lat = model.sample(batch, T, hw, steps=self.s["steps"], guidance=self.s["guidance"],
                                   shift=self.s["shift"], noise=torch.randn(latents.shape, generator=gen))
                video = model.decode(lat)[0].permute(0, 2, 3, 1).cpu().numpy()
                target = batch["target"][0].permute(0, 2, 3, 1).cpu().numpy()
                os.makedirs(out_dir, exist_ok=True)
                tag = f"{n:02d}_{name}_{str(batch['log_id'][0])[:8]}_{int(batch['moment'][0])}"
                write_video(os.path.join(out_dir, tag + ".mp4"), np.concatenate([target, video], axis=2))
        return {k: sums[k] / counts[k] for k in sorted(sums)}
