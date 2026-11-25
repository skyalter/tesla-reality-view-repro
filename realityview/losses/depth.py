"""Depth losses and depth metrics for training the backbone (no rendering involved).

All depths are z-depths: metres along the optical axis. Maps have shape (B, N, H, W): B samples of N views.
The losses take the predicted depth as its logarithm (`pred_log_z`), the form the network produces before any
clamping; the metrics take plain depths. Intrinsics use the integer pixel-centre convention of the processed
data (pixel (u, v) has its centre at (u, v)).

Every pixel of an input view falls into at most one of three classes (`pixel_classes`):
  lidar    a LiDAR point is there                  -> its depth is compared with the LiDAR depth
  sky      the sky score says sky                  -> its distance is pushed towards a far constant
  anchor   no LiDAR point and not sky              -> its depth is kept close to a frozen reference network
Ego-vehicle pixels, suspected see-through LiDAR points and uncertain sky scores are in no class.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

SEE_THROUGH = 4         # depth_flag bit: the LiDAR point is probably seen through a nearer surface


def pixel_classes(lidar_z, lidar_flag, ego_mask, sky=None, *, lidar_range=(0.5, 200.0), sky_low=0.3,
                  sky_high=0.5, sky_window=5) -> dict:
    """Boolean maps "lidar", "sky", "anchor", each (B, N, H, W).

    lidar_z     LiDAR depth, 0 where there is no point
    lidar_flag  flags of the LiDAR point (bit 4: suspected see-through)
    ego_mask    True on the ego vehicle (and on padding)
    sky         sky score in [0, 1], or None when sky scores are not available
    sky_low, sky_high   below sky_low a pixel is "not sky", from sky_high on it is "sky"; in between it is
                        left alone
    sky_window  a sky pixel is only pushed away if no LiDAR point lies in the window around it
    """
    has_point = lidar_z > 0
    usable = has_point & (lidar_z >= lidar_range[0]) & (lidar_z <= lidar_range[1])
    usable &= (lidar_flag.to(torch.int64) & SEE_THROUGH) == 0
    free = ~ego_mask
    if sky is None:
        not_sky = torch.ones_like(has_point)
        is_sky = torch.zeros_like(has_point)
    else:
        not_sky = sky < sky_low
        is_sky = sky >= sky_high
        shape = has_point.shape
        near_point = F.max_pool2d(has_point.reshape(-1, 1, *shape[-2:]).float(), sky_window, stride=1,
                                  padding=sky_window // 2).reshape(shape) > 0
        is_sky = is_sky & ~near_point
    return {"lidar": usable & free & not_sky, "sky": is_sky & free, "anchor": ~has_point & free & not_sky}


def ray_norms(K: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Length of the viewing ray with z = 1 through every pixel centre: distance = z-depth * ray norm.
    K (..., 3, 3) -> (..., H, W)."""
    v, u = torch.meshgrid(torch.arange(height, device=K.device, dtype=K.dtype),
                          torch.arange(width, device=K.device, dtype=K.dtype), indexing="ij")
    fx, fy = K[..., 0, 0, None, None], K[..., 1, 1, None, None]
    cx, cy = K[..., 0, 2, None, None], K[..., 1, 2, None, None]
    return torch.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the masked entries of every sample, then over the samples that have any. Zero (still attached
    to the graph) if no sample has one."""
    b = values.shape[0]
    values = torch.where(mask, values, torch.zeros_like(values)).reshape(b, -1)
    count = mask.reshape(b, -1).sum(1)
    per_sample = values.sum(1) / count.clamp(min=1)
    has_any = count > 0
    return (per_sample * has_any).sum() / has_any.sum().clamp(min=1)


def log_depth_error(pred_log_z: torch.Tensor, target_z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """|pred_log_z - log target| where the mask is set, 0 elsewhere. Targets outside the mask may be anything."""
    safe = torch.where(mask, target_z, torch.ones_like(target_z)).clamp(min=1e-6)
    return torch.where(mask, (pred_log_z - torch.log(safe)).abs(), torch.zeros_like(pred_log_z))


def lidar_depth_loss(pred_log_z, lidar_z, mask, conf=None, conf_lambda: float = 0.2) -> torch.Tensor:
    """Mean |log pred - log lidar| over LiDAR pixels. With a confidence map c >= 1 the per-pixel term becomes
    c * error - conf_lambda * log c, the form used by DUSt3R-style models."""
    error = log_depth_error(pred_log_z, lidar_z, mask)
    if conf is not None:
        error = conf * error - conf_lambda * torch.log(conf)
    return masked_mean(error, mask)


def sky_distance_loss(pred_log_z, ray_norm, mask, distance: float = 400.0) -> torch.Tensor:
    """Mean |log(distance of the pixel) - log(distance)| over sky pixels; distance = z * ray norm."""
    error = (pred_log_z + torch.log(ray_norm) - math.log(distance)).abs()
    return masked_mean(error, mask)


def align_reference(ref_log_z, lidar_z, lidar_mask, mode: str = "sample", min_points: int = 200,
                    max_depth: float = 80.0):
    """Shift the reference log-depth so that it agrees with LiDAR: one scale factor per sample (all views
    together) or per view, the median of lidar / reference over LiDAR pixels no farther than `max_depth`.

    Returns (aligned reference log-depth, valid) where valid is a boolean map that is False for samples or
    views with fewer than `min_points` such pixels; these have no trustworthy factor and are left as they
    are. mode "none" returns the reference unchanged."""
    if mode == "none":
        return ref_log_z, torch.ones_like(lidar_mask)
    if mode not in ("sample", "view"):
        raise ValueError(f"mode must be sample, view or none, got {mode}")
    b, n = ref_log_z.shape[:2]
    use = lidar_mask & (lidar_z <= max_depth)
    ratio = torch.log(torch.where(use, lidar_z, torch.ones_like(lidar_z)).clamp(min=1e-6)) - ref_log_z
    groups = 1 if mode == "sample" else n
    ratio, use = ratio.reshape(b, groups, -1), use.reshape(b, groups, -1)
    shift = torch.zeros(b, groups, device=ref_log_z.device, dtype=ref_log_z.dtype)
    ok = torch.zeros(b, groups, device=ref_log_z.device, dtype=torch.bool)
    for i in range(b):
        for j in range(groups):
            values = ratio[i, j][use[i, j]]
            if values.numel() >= min_points:
                shift[i, j] = values.median()
                ok[i, j] = True
    view_shape = (b, groups, 1, 1)
    return ref_log_z + shift.reshape(view_shape), ok.reshape(view_shape).expand_as(lidar_mask)


def anchor_loss(pred_log_z, aligned_ref_log_z, mask) -> torch.Tensor:
    """Mean |log pred - log reference| over the anchor pixels. The reference carries no gradient."""
    error = torch.where(mask, (pred_log_z - aligned_ref_log_z.detach()).abs(), torch.zeros_like(pred_log_z))
    return masked_mean(error, mask)


def median_scale(pred_z, target_z, mask) -> torch.Tensor:
    """(B,) median of target / pred over the masked pixels of each sample; 1 where a sample has none."""
    b = pred_z.shape[0]
    out = torch.ones(b, device=pred_z.device, dtype=pred_z.dtype)
    for i in range(b):
        m = mask[i]
        if m.any():
            out[i] = (target_z[i][m] / pred_z[i][m].clamp(min=1e-6)).median()
    return out


class DepthMetrics:
    """Running depth errors against LiDAR: AbsRel, delta < 1.25 and RMSE of log depth, overall, per distance
    band and per camera. Several named sets can be kept side by side (for example two LiDAR layers)."""

    def __init__(self, cam_names, bins=(0.0, 10.0, 20.0, 40.0, 80.0, 200.0)):
        self.cam_names = list(cam_names)
        self.edges = torch.tensor(bins[1:-1], dtype=torch.float64)
        self.bin_names = [f"{int(lo)}-{int(hi)}m" for lo, hi in zip(bins[:-1], bins[1:])]
        self.sums = {}

    def update(self, name: str, pred_z, gt_z, mask, cam_id) -> None:
        """pred_z, gt_z, mask: (B, N, H, W); cam_id: (B, N) index into cam_names."""
        if not mask.any():
            return
        pred = pred_z[mask].double().cpu()
        gt = gt_z[mask].double().cpu()
        cams = cam_id[:, :, None, None].expand_as(mask)[mask].cpu()
        rel = (pred - gt).abs() / gt
        hit = (torch.maximum(pred / gt, gt / pred) < 1.25).double()
        sq = (torch.log(pred) - torch.log(gt)) ** 2
        which_bin = torch.bucketize(gt, self.edges, right=True)
        state = self.sums.setdefault(name, {
            "bin": torch.zeros(4, len(self.bin_names), dtype=torch.float64),
            "cam": torch.zeros(4, len(self.cam_names), dtype=torch.float64)})
        for key, index, size in (("bin", which_bin, len(self.bin_names)), ("cam", cams, len(self.cam_names))):
            for row, weights in enumerate((torch.ones_like(rel), rel, hit, sq)):
                state[key][row] += torch.bincount(index, weights=weights, minlength=size)[:size]

    @staticmethod
    def _summary(rows: torch.Tensor) -> dict:
        n = rows[0].clamp(min=1)
        return {"absrel": rows[1] / n, "delta1": rows[2] / n, "rmse_log": torch.sqrt(rows[3] / n)}

    def compute(self) -> dict:
        """{"absrel_<name>": ..., "absrel_<name>/<band or camera>": ..., "pixels_<name>": ...} as floats."""
        out = {}
        for name, state in self.sums.items():
            total = state["bin"].sum(1, keepdim=True)
            for metric, value in self._summary(total).items():
                out[f"{metric}_{name}"] = float(value[0])
            out[f"pixels_{name}"] = float(total[0, 0])
            for key, labels in (("bin", self.bin_names), ("cam", self.cam_names)):
                summary = self._summary(state[key])
                for i, label in enumerate(labels):
                    if state[key][0, i] > 0:
                        for metric, value in summary.items():
                            out[f"{metric}_{name}/{label}"] = float(value[i])
                        out[f"pixels_{name}/{label}"] = float(state[key][0, i])
        return out


def overlap_consistency(pred_z, K, ref_T_cam, valid, slot, min_pixels: int = 200, tolerance: float = 0.05):
    """How well the depths of two cameras agree where they see the same place at the same moment.

    Every pixel of view i is lifted to 3D with its predicted depth and projected into every other view j of
    the same time slot. Where it lands inside j on a valid pixel, the depth it has there is compared with the
    depth j predicts at that pixel (nearest pixel).

    pred_z (N, H, W), K (N, 3, 3), ref_T_cam (N, 4, 4), valid (N, H, W) bool, slot (N,).
    Returns (median relative difference, share within `tolerance`, number of compared pixels), or None when
    fewer than `min_pixels` pixels overlap."""
    n, h, w = pred_z.shape
    dtype = torch.float64
    v, u = torch.meshgrid(torch.arange(h, device=pred_z.device, dtype=dtype),
                          torch.arange(w, device=pred_z.device, dtype=dtype), indexing="ij")
    pix = torch.stack([u, v, torch.ones_like(u)], -1).reshape(-1, 3)
    K = K.to(dtype)
    T = ref_T_cam.to(dtype)
    z = pred_z.to(dtype).reshape(n, -1)
    flat_valid = valid.reshape(n, -1)
    differences = []
    for i in range(n):
        points = (pix @ torch.linalg.inv(K[i]).T) * z[i][:, None]
        points = points[flat_valid[i]]
        if len(points) == 0:
            continue
        in_ref = points @ T[i, :3, :3].T + T[i, :3, 3]
        for j in range(n):
            if j == i or slot[j] != slot[i]:
                continue
            cam_T_ref = torch.linalg.inv(T[j])
            in_j = in_ref @ cam_T_ref[:3, :3].T + cam_T_ref[:3, 3]
            front = in_j[:, 2] > 0.1
            proj = in_j[front] @ K[j].T
            uj = torch.round(proj[:, 0] / proj[:, 2]).long()
            vj = torch.round(proj[:, 1] / proj[:, 2]).long()
            inside = (uj >= 0) & (uj < w) & (vj >= 0) & (vj < h)
            index = vj[inside] * w + uj[inside]
            keep = flat_valid[j][index]
            expected = in_j[front][inside][keep][:, 2]
            found = z[j][index][keep]
            differences.append((found - expected).abs() / expected)
    if not differences:
        return None
    differences = torch.cat(differences)
    if len(differences) < min_pixels:
        return None
    return float(differences.median()), float((differences < tolerance).double().mean()), int(len(differences))
