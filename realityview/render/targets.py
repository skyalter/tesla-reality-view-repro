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


def depth_jumps(depth: torch.Tensor) -> torch.Tensor:
    """Per pixel, the largest relative depth step to one of its 4 neighbours: max(d1, d2) / min(d1, d2) - 1.

    ``depth`` (N, H, W) > 0; returns (N, H, W). Both pixels of a step get its value. A plane seen at a grazing
    angle has small steps (the road under a camera 1.7 m above it: about 1 / (rows below the horizon)); the
    border between an object and what is behind it has large ones."""
    out = torch.zeros_like(depth)
    for dim in (1, 2):
        n = depth.shape[dim] - 1
        a, b = depth.narrow(dim, 0, n), depth.narrow(dim, 1, n)
        step = torch.maximum(a, b) / torch.minimum(a, b).clamp(min=1e-6) - 1.0
        lo, hi = out.narrow(dim, 0, n), out.narrow(dim, 1, n)
        lo.copy_(torch.maximum(lo, step))
        hi.copy_(torch.maximum(hi, step))
    return out


@torch.no_grad()
def cleanup_mask(g: Gaussians, keep: torch.Tensor, depth: torch.Tensor, sky: torch.Tensor | None = None, *,
                 sky_score: float | None = None, depth_jump: float | None = None,
                 max_size: float | None = None) -> torch.Tensor:
    """Which Gaussians of one sample to keep when the chase view is rendered: (G,) bool.

    The Gaussians are trained and checked from the input cameras only; from a camera 9 m up three kinds of them
    show as artefacts and can be left out (each test is off when its threshold is None):

        sky_score    Gaussians from pixels with a sky score >= ``sky_score`` (``sky`` (N, H, W), the raw
                     DA3METRIC score of the data set; 0.5 is the model's own sky mask)
        depth_jump   Gaussians from pixels where the predicted depth steps by more than ``depth_jump``
                     (relative, ``depth_jumps``) to a neighbour: they hang between an object and its background
        max_size     Gaussians whose longest axis is more than ``max_size`` times the size of their source pixel
                     at their depth (``footprint``): seen from close by they are large blurred blobs

    ``keep`` (N, H, W) bool are the pixels the Gaussians were made from, in the order of ``g``
    (``Stage1HeadTask.gaussian_masks``); ``depth`` (N, H, W) is the predicted depth of the input views.
    """
    if int(keep.sum()) != len(g):
        raise ValueError(f"keep has {int(keep.sum())} pixels, there are {len(g)} Gaussians")
    ok = torch.ones(len(g), dtype=torch.bool, device=g.means.device)
    if sky_score is not None:
        if sky is None:
            raise ValueError("sky_score needs the sky scores of the input views")
        ok &= sky[keep] < sky_score
    if depth_jump is not None:
        ok &= depth_jumps(depth.float())[keep] <= depth_jump
    if max_size is not None:
        ok &= g.scales.max(dim=-1).values <= max_size * g.footprint
    return ok


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


def overlay_mask(overlay, grow_px: int = 4) -> torch.Tensor:
    """Where a rendered car is: the pixels of an RGBA overlay ((H, W, 4) numpy, ``ego_overlay.render_mesh``)
    with any opacity, grown by ``grow_px`` pixels. (H, W) bool."""
    mask = (np.asarray(overlay)[..., 3] > 0).astype(np.uint8)
    if grow_px > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1)))
    return torch.from_numpy(mask > 0)


def soft_confidence(alpha: torch.Tensor, visibility: torch.Tensor, depth_consistency: torch.Tensor,
                    alpha_min: float = 0.5) -> torch.Tensor:
    """A weight for the stage-2 loss from the three stored factors:

        alpha * (0.5 + 0.5 * min(1, 2 V)) * sqrt(D),   0 where alpha < ``alpha_min``

    Compared with the plain product alpha * V * D: one confirming view is enough for full visibility (most
    surfaces are seen by one camera at two input moments only, so E[n_obs] is 1 and V = 0.5 for about half of
    the pixels), no confirmation halves the weight instead of zeroing it, and the depth term is as with a 10 %
    tolerance. Inputs are the factors as ``render_chase`` returns them (or the stored images / 255)."""
    v1 = (2.0 * visibility).clamp(max=1.0)
    conf = alpha * (0.5 + 0.5 * v1) * depth_consistency.clamp(min=0.0).sqrt()
    return torch.where(alpha >= alpha_min, conf, torch.zeros_like(conf))


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
