"""Chase-camera targets for stage 2: Gaussians of several moments rendered from a virtual camera behind and
above the ego vehicle, with a per-pixel confidence.

The chase view has no ground truth, so every target image comes with

    confidence = alpha * V * D

    alpha   accumulated opacity of the render
    V       visibility: min(1, E[n_obs] / 2), where n_obs of a Gaussian is the number of input views other than
            its source view in which it passes a depth-consistency check (``count_observations``); E[.] is the
            alpha-weighted mean over the Gaussians that make up a pixel
    D       depth consistency: exp(-sigma / (0.05 E[z])), sigma = sqrt(max(E[z^2] - E[z]^2, 0)), z = depth of
            each Gaussian in the chase camera; low where surfaces at different depths are blended

Everything here works on one sample (one set of ``Gaussians``) and needs no gradient. Conventions as in
``camera.py``: ``ref_T_cam`` camera -> reference, ``K`` integer pixel-centre convention, quaternions
(w, x, y, z), depth = z along the optical axis.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .camera import project, quat_mul, rotmat_to_quat
from .gaussians import FLAG_MOVER, Gaussians

# ego body box in the ego frame (x forward, y left, z up, origin at the rear-axle centre), metres; a rough
# outline of the Argoverse 2 vehicle (Ford Fusion: about 4.9 x 1.9 x 1.5 m, rear axle about 1.1 m from the back)
EGO_BOX = ((-1.3, 4.0), (-1.1, 1.1), (-0.4, 1.6))


@torch.no_grad()
def transform_gaussians(g: Gaussians, dst_T_src: torch.Tensor) -> Gaussians:
    """The same Gaussians in another frame: positions and orientations move, sizes, colours and opacities stay.

    ``dst_T_src`` (4, 4) maps the frame of ``g`` to the new one; it is applied in float64."""
    T = torch.as_tensor(dst_T_src, dtype=torch.float64, device=g.means.device)
    R, t = T[:3, :3], T[:3, 3]
    means = (g.means.double() @ R.T + t).to(g.means.dtype)
    q_R = rotmat_to_quat(R)
    quats = quat_mul(q_R.expand_as(g.quats), g.quats.double()).to(g.quats.dtype)
    return g.replace(means=means, quats=quats)


@torch.no_grad()
def count_observations(g: Gaussians, depth: torch.Tensor, K: torch.Tensor, ref_T_cam: torch.Tensor,
                       tolerance: float = 0.05) -> torch.Tensor:
    """How many input views other than its own see each Gaussian at the depth they predict: (G,) float.

    A Gaussian counts as observed by view v (v != g.view) when its centre projects inside the image of v, in
    front of the camera, and its depth z in v is within ``tolerance`` (relative) of the depth v predicts at the
    nearest pixel: |z - d| <= tolerance * d. ``depth`` (N, H, W), ``K`` (N, 3, 3), ``ref_T_cam`` (N, 4, 4) are
    the input views of the sample the Gaussians were made from (the same reference frame).
    """
    N, H, W = depth.shape
    counts = torch.zeros(len(g), dtype=torch.float32, device=g.means.device)
    means = g.means.float()
    for v in range(N):
        uv, z = project(means, ref_T_cam[v].float(), K[v].float())
        u = torch.round(uv[:, 0])
        r = torch.round(uv[:, 1])
        inside = (z > 0) & (u >= 0) & (u <= W - 1) & (r >= 0) & (r <= H - 1) & (g.view != v)
        ui = u.clamp(0, W - 1).long()
        ri = r.clamp(0, H - 1).long()
        d = depth[v].float()[ri, ui]
        ok = inside & (d > 0) & ((z - d).abs() <= tolerance * d)
        counts += ok.float()
    return counts


def merge_moments(parts):
    """Merge the Gaussians of several moments into one frame.

    ``parts``: iterable of (Gaussians, n_obs (G,), dst_T_src (4, 4) or None for the identity, is_current).
    The Gaussians of the current moment are all kept; of the other moments only the static ones (no
    FLAG_MOVER): moving objects are taken from the current moment only.
    Returns (Gaussians, n_obs, current (G,) bool).
    """
    gs, counts, current = [], [], []
    for g, n_obs, dst_T_src, is_current in parts:
        if dst_T_src is not None:
            g = transform_gaussians(g, dst_T_src)
        if not is_current:
            static = (g.flags & FLAG_MOVER) == 0
            g = Gaussians(**{k: v[static] for k, v in g.__dict__.items()})
            n_obs = n_obs[static]
        gs.append(g)
        counts.append(n_obs)
        current.append(torch.full((len(g),), bool(is_current), dtype=torch.bool, device=g.means.device))
    return Gaussians.cat(gs), torch.cat(counts), torch.cat(current)


@torch.no_grad()
def ego_box_mask(ref_T_ego: torch.Tensor, ref_T_cam: torch.Tensor, K: torch.Tensor, height: int, width: int,
                 box=EGO_BOX) -> torch.Tensor:
    """Where the ego vehicle is in a camera: the convex hull of the 8 projected corners of ``box`` (ego frame,
    ((x0, x1), (y0, y1), (z0, z1))), filled. (H, W) bool on the device of ``K``. Corners behind the camera are
    not handled (the chase camera sees the whole car)."""
    (x0, x1), (y0, y1), (z0, z1) = box
    corners = torch.tensor([[x, y, z, 1.0] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)],
                           dtype=torch.float64)
    ref = corners @ ref_T_ego.double().cpu().T
    uv, z = project(ref[:, :3], ref_T_cam.double().cpu(), K.double().cpu())
    mask = np.zeros((height, width), np.uint8)
    if bool((z > 0).all()):
        shift = 4                                           # sub-pixel precision for cv2
        pts = np.round(uv.numpy() * (1 << shift)).astype(np.int64)
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1, lineType=cv2.LINE_8, shift=shift)
    return torch.from_numpy(mask > 0).to(K.device)


@torch.no_grad()
def render_chase(g: Gaussians, n_obs: torch.Tensor, ref_T_cam: torch.Tensor, K: torch.Tensor, height: int,
                 width: int, background: torch.Tensor | None = None, mode: str = "antialiased", *,
                 near: float = 0.1, far: float = 1.0e4, alpha_min: float = 1.0e-3, rel_sigma: float = 0.05) -> dict:
    """Render one chase camera (``ref_T_cam`` (4, 4), ``K`` (3, 3)) with its confidence.

    Returns, all (H, W) except rgb (3, H, W):
        rgb                composited on ``background`` (3,) (None = black)
        alpha              accumulated opacity
        visibility         V = min(1, E[n_obs] / 2)
        depth_consistency  D = exp(-sigma / (rel_sigma * E[z])), sigma^2 = E[z^2] - E[z]^2 (z in this camera)
        confidence         alpha * V * D
        depth              expected z
        mover              E[FLAG_MOVER]: how much of a pixel comes from Gaussians of moving objects
    V and D are 0 where alpha < ``alpha_min``.
    """
    from .rasterize import _rasterize

    cam_T_ref = torch.linalg.inv(ref_T_cam.float())
    z = g.means.float() @ cam_T_ref[2, :3] + cam_T_ref[2, 3]                   # depth in this camera
    mover = ((g.flags & FLAG_MOVER) > 0).float()
    features = torch.cat([g.colors.float(), n_obs.float()[:, None], (z * z)[:, None], mover[:, None]], dim=-1)
    out, alpha = _rasterize(g.means, g.quats, g.scales, g.opacities, features, ref_T_cam[None], K[None],
                            height, width, mode=mode, near=near, far=far, chunk=None)
    out, alpha = out[0], alpha[0, ..., 0]
    hit = alpha >= alpha_min
    a = alpha.clamp(min=alpha_min)
    rgb = out[..., 0:3]
    if background is not None:
        rgb = rgb + (1.0 - alpha)[..., None] * background.to(rgb)
    ez = out[..., 6]                                                            # already divided by alpha
    ez2 = out[..., 4] / a
    sigma = torch.sqrt((ez2 - ez * ez).clamp(min=0.0))
    vis = torch.where(hit, (out[..., 3] / a / 2.0).clamp(max=1.0), torch.zeros_like(alpha))
    dcons = torch.where(hit, torch.exp(-sigma / (rel_sigma * ez.clamp(min=1e-6))), torch.zeros_like(alpha))
    return {"rgb": rgb.permute(2, 0, 1), "alpha": alpha, "visibility": vis, "depth_consistency": dcons,
            "confidence": alpha * vis * dcons, "depth": ez,
            "mover": torch.where(hit, (out[..., 5] / a).clamp(0.0, 1.0), torch.zeros_like(alpha))}
