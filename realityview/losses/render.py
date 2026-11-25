"""Losses and measures that compare a rendered image with a photo, and a rendered depth with LiDAR.

All images are (M, 3, H, W) in 0..1 and masks (M, H, W) bool; a mask selects the pixels that are compared.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..data.config import FLAG_SEETHROUGH, FLAG_SYNC_DYNAMIC


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the selected pixels and the 3 channels; zero (with a graph) if none is selected."""
    m = mask[:, None].to(pred.dtype)
    return ((pred - target) ** 2 * m).sum() / (3.0 * m.sum()).clamp(min=1.0)


def psnr(mse: float) -> float:
    """PSNR in dB of images in 0..1."""
    return -10.0 * math.log10(max(float(mse), 1e-10))


class PerceptualLoss(nn.Module):
    """LPIPS (VGG by default) between a render and a photo.

    Pixels outside the mask are replaced by the photo in the render, so they add nothing. The network is
    frozen; it is not part of the model and is not saved in checkpoints. Needs the ``lpips`` package and, on
    first use, the ImageNet VGG weights of torchvision (downloaded to the torch hub cache).
    """

    def __init__(self, net: str = "vgg"):
        super().__init__()
        import lpips

        self.net_name = net
        self.version = getattr(lpips, "__version__", "?")
        self.net = lpips.LPIPS(net=net, verbose=False)
        self.net.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean LPIPS over the M images (inputs 0..1)."""
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred, target = pred.float(), target.float()
            if mask is not None:
                pred = torch.where(mask[:, None], pred, target)
            return self.net(pred, target, normalize=True).mean()


def rendered_depth_mask(depth: torch.Tensor, alpha: torch.Tensor, lidar_z: torch.Tensor, lidar_flag: torch.Tensor,
                        region: torch.Tensor, *, alpha_min: float = 0.5, lidar_range=(0.5, 200.0)) -> torch.Tensor:
    """Pixels where a rendered depth is compared with LiDAR: inside ``region``, rendered opacity above
    ``alpha_min``, a LiDAR point in range that is neither suspected see-through nor a synchronised point on
    a moving object (those belong to the object's position at the input moment)."""
    flag = lidar_flag.to(torch.int64)
    return (region & (alpha > alpha_min) & (depth > 0) & (lidar_z >= lidar_range[0]) & (lidar_z <= lidar_range[1])
            & ((flag & (FLAG_SEETHROUGH | FLAG_SYNC_DYNAMIC)) == 0))


def rendered_depth_loss(depth: torch.Tensor, lidar_z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean |log rendered z - log LiDAR z| over the mask (gsplat's expected depth is already divided by the
    accumulated weight)."""
    err = (torch.log(depth.clamp(min=1e-3)) - torch.log(lidar_z.clamp(min=1e-3))).abs()
    m = mask.to(err.dtype)
    return (err * m).sum() / m.sum().clamp(min=1.0)
