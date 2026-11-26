"""A stand-in for the video autoencoder with the same shapes (4x in time, 8x in space, 16 channels) and no
weights: block averages of the image, repeated over the channels. For tests on the CPU only."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyVAE(nn.Module):
    time_stride, space_stride, channels = 4, 8, 16

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """uint8 (B, T, 3, H, W), T = 1 + 4k -> (B, 16, 1 + k, H / 8, W / 8)."""
        B, T = video.shape[:2]
        x = F.avg_pool2d(video.float().div(127.5).sub(1.0).flatten(0, 1), 8).unflatten(0, (B, T))   # (B, T, 3, h, w)
        first, rest = x[:, :1], x[:, 1:].unflatten(1, (-1, 4)).mean(dim=2)
        x = torch.cat([first, rest], dim=1).permute(0, 2, 1, 3, 4)                                   # (B, 3, F, h, w)
        return torch.cat([x] * 5 + [x[:, :1]], dim=1)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        x = latents[:, :3].permute(0, 2, 1, 3, 4)                                                    # (B, F, 3, h, w)
        x = torch.cat([x[:, :1], x[:, 1:].repeat_interleave(4, dim=1)], dim=1)
        B, T = x.shape[:2]
        x = F.interpolate(x.flatten(0, 1), scale_factor=8, mode="nearest").unflatten(0, (B, T))
        return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
