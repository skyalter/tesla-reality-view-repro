"""The conditions of the stage-2 diffusion model: camera feature tokens with their geometry, and the rays of
the target camera.

Feature tokens. Every input image gives the backbone's four feature levels (each ``patch_hw`` tokens of
``feat_dim`` channels). Each level is averaged over 2 x 2 patches (28 x 37 patches -> 14 x 19 tokens per
image, ``pool_features``, no parameters), the levels are concatenated, normalised and projected to the width of
the diffusion backbone. A learned embedding of the camera and of the input slot (the moment itself / 0.5 s
earlier) is added. Averaging comes first on purpose: the un-pooled features of one sample are several GB, and
whatever feeds the first trainable layer has to be kept for the backward pass.

Geometry. For every latent frame of the target video, every token gets an encoding of where it is as seen from
the target camera of that frame: its viewing ray, the 3-D point given by the depth head, where that point lands
in the target image, and the time difference. The same tokens therefore get a different geometry for every
latent frame. ``use_depth=False`` leaves only the ray and the time (nothing from the depth head).

Target rays. One Pluecker ray per 16 x 16 pixel block of the target image, in the ego frame; added to the
patch embedding of the diffusion backbone. Its last layer starts at zero.

Conventions as everywhere in this project: poses are camera -> reference, OpenCV camera axes, metres;
intrinsics have integer pixel centres.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MERGE = 2                   # tokens are merged MERGE x MERGE
GEOM_SCALE_M = 20.0         # lengths are divided by this before they are encoded
POINT_FREQS = 6
PIXEL_FREQS = 6
TIME_FREQS = 4
DIR_FREQS = 4


def fourier(x: torch.Tensor, n_freqs: int) -> torch.Tensor:
    """(..., C) -> (..., C * (1 + 2 n)): x, sin(2^k pi x), cos(2^k pi x) for k < n."""
    freqs = (2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)) * math.pi
    ang = x.unsqueeze(-1) * freqs
    return torch.cat([x, ang.sin().flatten(-2), ang.cos().flatten(-2)], dim=-1)


GEOM_DIM = 6 + 3 * (1 + 2 * POINT_FREQS) + 2 * (1 + 2 * PIXEL_FREQS) + 3 + (1 + 2 * TIME_FREQS)
TARGET_RAY_DIM = 6 + 3 * 2 * DIR_FREQS


def merged_grid(patch_hw) -> tuple[int, int]:
    return (math.ceil(patch_hw[0] / MERGE), math.ceil(patch_hw[1] / MERGE))


def token_centres(image_hw, patch_hw, device=None, dtype=torch.float32):
    """Pixel coordinates (u, v) of the centre of every merged token and the pixel block it covers.

    A merged token covers MERGE x MERGE patches; the last row / column can reach beyond the image (37 patches
    are padded to 38), its centre is then the centre of the part inside the image. Returns (gh * gw, 2)."""
    H, W = image_hw
    gh, gw = merged_grid(patch_hw)
    ph, pw = H / patch_hw[0] * MERGE, W / patch_hw[1] * MERGE
    v0 = torch.arange(gh, device=device, dtype=dtype) * ph
    u0 = torch.arange(gw, device=device, dtype=dtype) * pw
    v = (v0 + torch.clamp(v0 + ph, max=H) - 1) / 2
    u = (u0 + torch.clamp(u0 + pw, max=W) - 1) / 2
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([uu, vv], dim=-1).reshape(-1, 2)


def token_log_depth(depth: torch.Tensor, patch_hw) -> torch.Tensor:
    """Mean log depth of every merged token: depth (..., H, W) metres -> (..., gh * gw)."""
    H, W = depth.shape[-2:]
    gh, gw = merged_grid(patch_hw)
    kh, kw = round(H / patch_hw[0] * MERGE), round(W / patch_hw[1] * MERGE)
    x = depth.clamp(min=1e-3).log().reshape(-1, 1, H, W)
    x = F.pad(x, (0, gw * kw - W, 0, gh * kh - H), mode="replicate")
    return F.avg_pool2d(x, (kh, kw)).reshape(*depth.shape[:-2], gh * gw)


@torch.no_grad()
def token_geometry(log_depth, K, clip_T_cam, src_time, tgt_clip_T_cam, tgt_K, tgt_hw, tgt_time, moment_of_frame,
                   image_hw, patch_hw, use_depth: bool = True, scale: float = GEOM_SCALE_M) -> torch.Tensor:
    """The geometry encoding of every token for every latent frame, before the learned layers.

        log_depth       (M, N, L) mean log depth of the tokens (``token_log_depth``)
        K               (M, N, 3, 3) intrinsics of the input images
        clip_T_cam      (M, N, 4, 4) input camera -> clip reference frame
        src_time        (M, N) time of the input images, seconds
        tgt_clip_T_cam  (F, 4, 4) target camera -> clip reference frame, one per latent frame
        tgt_K           (3, 3) target intrinsics at ``tgt_hw`` = (height, width)
        tgt_time        (F,) time of the target frames, seconds (same clock as src_time)
        moment_of_frame (F,) index of the conditioning moment of every latent frame
    Returns (F, N * L, GEOM_DIM), single precision. M = conditioning moments, N = views, L = tokens per view.
    """
    M, N, L = log_depth.shape
    dev = log_depth.device
    uv = token_centres(image_hw, patch_hw, dev)                                    # (L, 2)
    fx, fy, cx, cy = K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]         # (M, N)
    rays = torch.stack([(uv[:, 0] - cx[..., None]) / fx[..., None], (uv[:, 1] - cy[..., None]) / fy[..., None],
                        torch.ones(M, N, L, device=dev)], dim=-1)                   # (M, N, L, 3), z = 1
    pts_cam = rays * log_depth.exp().unsqueeze(-1)
    tgt_T_clip = torch.linalg.inv(tgt_clip_T_cam.double())                         # (F, 4, 4)
    T = (tgt_T_clip[:, None] @ clip_T_cam.double()[moment_of_frame]).float()        # (F, N, 4, 4) input cam -> target cam
    R, t = T[..., :3, :3], T[..., :3, 3]
    rays_f, pts_f = rays[moment_of_frame], pts_cam[moment_of_frame]                 # (F, N, L, 3)
    d = F.normalize(rays_f @ R.transpose(-1, -2), dim=-1)                           # ray direction in the target camera
    o = (t / scale)[:, :, None, :].expand_as(d)                                     # ray origin
    pluecker = torch.cat([d, torch.linalg.cross(o, d, dim=-1)], dim=-1)
    p = pts_f @ R.transpose(-1, -2) + t[:, :, None, :]                              # point in the target camera
    z = p[..., 2]
    front = z > 0.1
    zs = z.clamp(min=0.1)
    Ht, Wt = tgt_hw
    un = (tgt_K[0, 0] * p[..., 0] / zs + tgt_K[0, 2]) / (Wt - 1) * 2 - 1
    vn = (tgt_K[1, 1] * p[..., 1] / zs + tgt_K[1, 2]) / (Ht - 1) * 2 - 1
    inside = front & (un.abs() <= 1) & (vn.abs() <= 1)
    pix = torch.stack([un, vn], dim=-1).clamp(-2, 2) * front.unsqueeze(-1)
    flags = torch.stack([zs.log() / 5.0 * front, inside.float(), front.float()], dim=-1)
    point = (p / scale).clamp(-25, 25)
    from_depth = torch.cat([fourier(point / 4.0, POINT_FREQS), fourier(pix / 2.0, PIXEL_FREQS), flags], dim=-1)
    if not use_depth:
        from_depth = torch.zeros_like(from_depth)
    dt = (src_time[moment_of_frame] - tgt_time[:, None])[:, :, None, None].expand(-1, -1, L, 1)
    out = torch.cat([pluecker, from_depth, fourier(dt, TIME_FREQS)], dim=-1)
    return out.reshape(out.shape[0], N * L, GEOM_DIM)


@torch.no_grad()
def target_rays(tgt_K, ego_T_cam, tgt_hw, patch_px: int = 16, scale: float = GEOM_SCALE_M) -> torch.Tensor:
    """Pluecker ray of the centre of every ``patch_px`` block of the target image, in the ego frame.

    tgt_K (B, 3, 3), ego_T_cam (B, 4, 4) -> (B, (H / patch) * (W / patch), TARGET_RAY_DIM)."""
    H, W = tgt_hw
    dev = tgt_K.device
    v = torch.arange(H // patch_px, device=dev, dtype=torch.float32) * patch_px + (patch_px - 1) / 2
    u = torch.arange(W // patch_px, device=dev, dtype=torch.float32) * patch_px + (patch_px - 1) / 2
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    uu, vv = uu.reshape(-1), vv.reshape(-1)
    rays = torch.stack([(uu - tgt_K[:, 0, 2, None]) / tgt_K[:, 0, 0, None],
                        (vv - tgt_K[:, 1, 2, None]) / tgt_K[:, 1, 1, None],
                        torch.ones(tgt_K.shape[0], uu.numel(), device=dev)], dim=-1)
    d = F.normalize(rays @ ego_T_cam[:, :3, :3].transpose(-1, -2), dim=-1)
    o = (ego_T_cam[:, :3, 3] / scale)[:, None, :].expand_as(d)
    ang = d.unsqueeze(-1) * ((2.0 ** torch.arange(DIR_FREQS, device=dev)) * math.pi)
    return torch.cat([d, torch.linalg.cross(o, d, dim=-1), ang.sin().flatten(-2), ang.cos().flatten(-2)], dim=-1)


@torch.no_grad()
def pool_features(feats, patch_hw, dtype=None) -> torch.Tensor:
    """Backbone feature levels, each (M, N, h * w, D) -> (M, N, L, levels * D): the mean over every
    MERGE x MERGE block of patches (blocks at the border are smaller), levels concatenated."""
    h, w = patch_hw
    out = []
    for f in feats:
        M, N, _, D = f.shape
        x = f.reshape(M * N, h, w, D).permute(0, 3, 1, 2).float()
        x = F.avg_pool2d(x, MERGE, ceil_mode=True, count_include_pad=False)
        out.append(x.flatten(2).transpose(1, 2).reshape(M, N, -1, D).to(dtype or f.dtype))
    return torch.cat(out, dim=-1)


class Conditioner(nn.Module):
    """The trainable part of the conditions.

    Args:
        feat_dim: channels of one backbone feature level (2 x embedding width of the backbone)
        n_levels: number of feature levels (4)
        dim: width of the diffusion backbone
        patch_hw: backbone patches per image (28, 37)
    """

    def __init__(self, feat_dim: int, n_levels: int = 4, dim: int = 1536, patch_hw=(28, 37), n_cams: int = 7,
                 n_slots: int = 2, use_depth: bool = True):
        super().__init__()
        self.patch_hw = tuple(patch_hw)
        self.grid = merged_grid(patch_hw)
        self.tokens_per_view = self.grid[0] * self.grid[1]
        self.use_depth = bool(use_depth)
        self.feat_norm = nn.LayerNorm(feat_dim * n_levels)
        self.feat_proj = nn.Linear(feat_dim * n_levels, dim)
        self.cam_embed = nn.Parameter(torch.zeros(n_cams, dim))
        self.slot_embed = nn.Parameter(torch.zeros(n_slots, dim))
        self.geom = nn.Sequential(nn.Linear(GEOM_DIM, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.out_norm = nn.LayerNorm(dim)
        self.null_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.target = nn.Sequential(nn.Linear(TARGET_RAY_DIM, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.target[-1].weight)
        nn.init.zeros_(self.target[-1].bias)

    def feature_tokens(self, pooled, cam_id, slot) -> torch.Tensor:
        """pooled (M, N, L, n_levels * feat_dim) from ``pool_features``; cam_id, slot (M, N) -> (M, N * L, dim)."""
        M, N, L, _ = pooled.shape
        x = self.feat_proj(self.feat_norm(pooled.to(self.feat_proj.weight.dtype)))
        x = x + self.cam_embed[cam_id][:, :, None, :].to(x.dtype) + self.slot_embed[slot][:, :, None, :].to(x.dtype)
        return x.reshape(M, N * L, -1)

    def frame_tokens(self, tokens, geometry, moment_of_frame) -> torch.Tensor:
        """tokens (M, N * L, dim) of one sample, geometry (F, N * L, GEOM_DIM) -> (F, N * L, dim)."""
        x = tokens[moment_of_frame] + self.geom(geometry.to(tokens.dtype))
        return self.out_norm(x)

    def target_embedding(self, rays, n_frames: int) -> torch.Tensor:
        """rays (B, hw, TARGET_RAY_DIM) -> (B, n_frames * hw, dim): the same for every latent frame."""
        x = self.target(rays.to(self.target[0].weight.dtype))
        return x.unsqueeze(1).expand(-1, n_frames, -1, -1).reshape(x.shape[0], -1, x.shape[-1])
