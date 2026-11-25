"""One 3D Gaussian per input pixel, in the reference frame.

    position   mu = R (K^-1 (u, v, 1)^T z) + t        R, t from ref_T_cam; z is the predicted depth along the
                                                     optical axis; K^-1 (u, v, 1) is not normalised
    size       (lo + (hi - lo) sigmoid(a)) * footprint, capped at ``max_scale_rel`` * distance to the camera,
               where footprint = z * 0.5 (1/fx + 1/fy) is the size of one pixel at that depth
    rotation   (camera -> reference rotation) * normalize(q), quaternions (w, x, y, z)
    colour     sigmoid(logit(clip(I, 0.01, 0.99)) + f)   I = the input pixel, f a residual
    opacity    sigmoid(o + opacity_bias)

The 11 raw channels per pixel are [a (3), q (4), f (3), o (1)]. With a = -2 (size about 1.04 pixel
footprints), q = (1, 0, 0, 0), f = 0 and o = 0 a Gaussian sits on the predicted surface, has the colour of
its pixel and opacity sigmoid(-2) = 0.12.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

from .camera import pixel_rays, quat_mul, rotmat_to_quat

RAW_CHANNELS = 11
FLAG_MOVER = 1          # the Gaussian comes from a pixel on a moving object or a person


@dataclass
class Gaussians:
    """The Gaussians of one sample. G = number of Gaussians."""
    means: torch.Tensor         # (G, 3) reference frame, metres
    quats: torch.Tensor         # (G, 4) (w, x, y, z), reference frame
    scales: torch.Tensor        # (G, 3) metres (standard deviations)
    opacities: torch.Tensor     # (G,)
    colors: torch.Tensor        # (G, 3) 0..1
    footprint: torch.Tensor     # (G,) size of the source pixel at the Gaussian's depth, metres (no gradient)
    flags: torch.Tensor         # (G,) uint8, FLAG_* bits
    view: torch.Tensor          # (G,) int64, index of the input view the Gaussian comes from

    def __len__(self) -> int:
        return self.means.shape[0]

    def detach(self) -> "Gaussians":
        return Gaussians(**{f.name: getattr(self, f.name).detach() for f in fields(self)})

    def replace(self, **changes) -> "Gaussians":
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values.update(changes)
        return Gaussians(**values)

    @staticmethod
    def cat(parts) -> "Gaussians":
        return Gaussians(**{f.name: torch.cat([getattr(p, f.name) for p in parts]) for f in fields(Gaussians)})


def unproject(depth: torch.Tensor, K: torch.Tensor, ref_T_cam: torch.Tensor) -> torch.Tensor:
    """Depth maps (N, H, W) (z along the optical axis) -> points (N, H, W, 3) in the reference frame."""
    rays = pixel_rays(K, depth.shape[-2], depth.shape[-1])
    pts = rays * depth[..., None]
    R, t = ref_T_cam[:, :3, :3], ref_T_cam[:, :3, 3]
    return torch.einsum("nij,nhwj->nhwi", R, pts) + t[:, None, None, :]


def assemble(raw: torch.Tensor, images: torch.Tensor, depth: torch.Tensor, K: torch.Tensor,
             ref_T_cam: torch.Tensor, keep: torch.Tensor, mover: torch.Tensor | None = None, *,
             scale_range=(0.1, 8.0), max_scale_rel: float = 0.02, opacity_bias: float = -2.0) -> Gaussians:
    """Gaussians of one sample from the per-pixel outputs of its N input views.

    Args:
        raw: (N, 11, H, W) appearance outputs [a (3), q (4), f (3), o (1)].
        images: (N, 3, H, W) float 0..1, the input images.
        depth: (N, H, W) predicted z in metres (carries the gradient to the depth branch).
        K: (N, 3, 3) integer pixel-centre intrinsics; ref_T_cam: (N, 4, 4) camera -> reference.
        keep: (N, H, W) bool, the pixels that produce a Gaussian.
        mover: (N, H, W) bool or None, pixels on moving objects or persons (sets FLAG_MOVER).
    """
    if raw.shape[1] != RAW_CHANNELS:
        raise ValueError(f"raw must have {RAW_CHANNELS} channels, got {raw.shape[1]}")
    N, _, H, W = raw.shape
    K, ref_T_cam = K.float(), ref_T_cam.float()
    rays = pixel_rays(K, H, W)                                           # (N, H, W, 3)
    R, t = ref_T_cam[:, :3, :3], ref_T_cam[:, :3, 3]
    means = torch.einsum("nij,nhwj->nhwi", R, rays * depth[..., None]) + t[:, None, None, :]

    z = depth.detach()
    footprint = z * (0.5 * (1.0 / K[:, 0, 0] + 1.0 / K[:, 1, 1]))[:, None, None]      # metres per pixel
    distance = z * rays.norm(dim=-1)
    lo, hi = scale_range
    scales = (lo + (hi - lo) * torch.sigmoid(raw[:, 0:3])) * footprint[:, None]
    scales = torch.minimum(scales, (max_scale_rel * distance)[:, None])               # (N, 3, H, W)

    q_cam = F.normalize(raw[:, 3:7].permute(0, 2, 3, 1), dim=-1)                       # (N, H, W, 4)
    quats = quat_mul(rotmat_to_quat(R)[:, None, None, :], q_cam)

    colors = torch.sigmoid(torch.logit(images.clamp(0.01, 0.99)) + raw[:, 7:10])     # (N, 3, H, W)
    opacities = torch.sigmoid(raw[:, 10] + opacity_bias)                              # (N, H, W)

    flags = torch.zeros((N, H, W), dtype=torch.uint8, device=raw.device)
    if mover is not None:
        flags = flags | (mover.to(torch.uint8) * FLAG_MOVER)
    view = torch.arange(N, device=raw.device)[:, None, None].expand(N, H, W)
    return Gaussians(means=means[keep], quats=quats[keep], scales=scales.permute(0, 2, 3, 1)[keep],
                     opacities=opacities[keep], colors=colors.permute(0, 2, 3, 1)[keep],
                     footprint=footprint[keep], flags=flags[keep], view=view[keep])
