"""Rendering of ``Gaussians`` with gsplat.

``render`` returns colour, expected depth, opacity and the "mover" channel for a set of target cameras;
``coverage`` tells which target pixels are hit by anything the inputs have seen, independent of the learned
opacities and sizes.

Cameras are given the way the rest of the code has them: ``ref_T_cam`` (camera -> reference) and ``K`` in the
integer pixel-centre convention. gsplat wants world -> camera matrices and pixel centres at j + 0.5; both
conversions happen here and nowhere else.
"""
from __future__ import annotations

import torch

from .camera import half_pixel_K
from .gaussians import FLAG_MOVER, Gaussians


def _rasterize(means, quats, scales, opacities, features, ref_T_cam, K, height, width, *, mode, near, far, chunk):
    from gsplat import rasterization

    viewmats = torch.linalg.inv(ref_T_cam.float())
    Ks = half_pixel_K(K.float())
    C = viewmats.shape[0]
    step = C if not chunk else chunk
    colors, alphas = [], []
    for s in range(0, C, step):
        # no ``backgrounds``: gsplat 1.5.3 rejects them in packed mode; the caller composites instead
        out, alpha, _ = rasterization(means, quats, scales, opacities, features, viewmats[s:s + step],
                                      Ks[s:s + step], width, height, near_plane=near, far_plane=far,
                                      render_mode="RGB+ED", rasterize_mode=mode, packed=True)
        colors.append(out)
        alphas.append(alpha)
    return torch.cat(colors), torch.cat(alphas)


def render(g: Gaussians, ref_T_cam: torch.Tensor, K: torch.Tensor, height: int, width: int, *,
           background: torch.Tensor | None = None, mode: str = "antialiased", near: float = 0.1,
           far: float = 1.0e4, chunk: int | None = 7) -> dict:
    """Render the Gaussians of one sample into C target cameras.

    Args:
        ref_T_cam: (C, 4, 4) target camera -> reference; K: (C, 3, 3) integer pixel-centre intrinsics.
        background: (3,) colour behind the Gaussians (None = black).
        mode: gsplat rasterize mode, "antialiased" or "classic".
        chunk: cameras per rasterization call (memory).
    Returns:
        rgb (C, 3, H, W); depth (C, H, W) expected z = sum(w z) / sum(w) (gsplat "ED": already divided by
        the accumulated weight, 0 where nothing was hit); alpha (C, H, W) accumulated opacity;
        mover (C, H, W) the rendered FLAG_MOVER channel (how much of a pixel comes from Gaussians of moving
        objects or persons of the input moment; no gradient use intended).
    """
    mover = ((g.flags & FLAG_MOVER) > 0).to(g.colors.dtype)[:, None]
    features = torch.cat([g.colors, mover], dim=-1)
    out, alpha = _rasterize(g.means, g.quats, g.scales, g.opacities, features, ref_T_cam, K, height, width,
                            mode=mode, near=near, far=far, chunk=chunk)
    rgb = out[..., 0:3]
    if background is not None:
        rgb = rgb + (1.0 - alpha) * background
    return {"rgb": rgb.permute(0, 3, 1, 2), "mover": out[..., 3], "depth": out[..., 4], "alpha": alpha[..., 0]}


@torch.no_grad()
def coverage(g: Gaussians, ref_T_cam: torch.Tensor, K: torch.Tensor, height: int, width: int, *,
             footprint: float = 1.0, near: float = 0.1, far: float = 1.0e4, chunk: int | None = 7) -> torch.Tensor:
    """How much of each target pixel is covered by surface the inputs have seen: (C, H, W) in 0..1.

    Every Gaussian is replaced by an opaque isotropic one whose size is ``footprint`` times the size of its
    source pixel at its depth, so the result depends on the predicted positions only, not on the learned
    opacities and sizes. Dense surface gives about 1; a region no input pixel projects into gives 0.
    """
    scales = (g.footprint * footprint)[:, None].expand(-1, 3)
    ones = torch.ones_like(g.footprint)
    _, alpha = _rasterize(g.means, g.quats, scales, ones, ones[:, None], ref_T_cam, K, height, width,
                          mode="classic", near=near, far=far, chunk=chunk)
    return alpha[..., 0]
