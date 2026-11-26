"""The loss of the stage-2 diffusion model: a weighted mean squared error in latent space.

The targets rendered from the Gaussian model are only partly trustworthy, so every latent position has a
weight derived from the confidence of the pixels it covers; photos have weight 1. The ego vehicle is never
supervised (weight 0).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_pool(x: torch.Tensor, stride: int = 4) -> torch.Tensor:
    """Per-frame maps (B, T, h, w) with T = 1 + stride * k -> per-latent-frame maps (B, 1 + k, h, w):
    the first frame alone, then the mean of every ``stride`` frames (as the video autoencoder groups them)."""
    B, T, h, w = x.shape
    if (T - 1) % stride:
        raise ValueError(f"{T} frames do not fit 1 + {stride} k")
    rest = x[:, 1:].reshape(B, (T - 1) // stride, stride, h, w).mean(dim=2)
    return torch.cat([x[:, :1], rest], dim=1)


def spatial_pool(x: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """(B, T, H, W) -> (B, T, H / factor, W / factor), mean of every block."""
    B, T, H, W = x.shape
    return F.avg_pool2d(x.reshape(B * T, 1, H, W), factor).reshape(B, T, H // factor, W // factor)


def confidence_weight(conf: torch.Tensor, supervise: torch.Tensor, w_min: float = 0.05, c_ref: float = 0.5) -> torch.Tensor:
    """Loss weight of every latent position.

        conf        (B, F, h, w) confidence in 0..1, already pooled to the latent grid
        supervise   (B, F, h, w) share of the position that is supervised at all (0 on the ego vehicle)
    weight = supervise * (w_min + (1 - w_min) * min(1, conf / c_ref)).  A photo has conf = 1 everywhere.
    The floor ``w_min`` keeps a little supervision where the rendering is not trusted, so that the model is
    not left completely free there."""
    return supervise * (w_min + (1.0 - w_min) * (conf / c_ref).clamp(max=1.0))


def weighted_flow_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
                       sample_weight: torch.Tensor | None = None):
    """pred, target (B, C, F, h, w); weight (B, F, h, w); sample_weight (B,) or None.

    Per sample: sum(weight * squared error) / (sum(weight) * C); then the (weighted) mean over the batch.
    Returns (loss, per-sample losses detached)."""
    err = (pred.float() - target.float()).pow(2).mean(dim=1)            # (B, F, h, w)
    w = weight.float()
    per = (err * w).flatten(1).sum(dim=1) / w.flatten(1).sum(dim=1).clamp(min=1e-6)
    if sample_weight is None:
        return per.mean(), per.detach()
    return (per * sample_weight.float()).mean(), per.detach()
