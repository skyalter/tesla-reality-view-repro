"""Camera helpers shared by the Gaussian head and the renderer (torch).

Conventions, the same as the data reader (``realityview/data/dataset.py``):

- ``K`` (..., 3, 3): pinhole intrinsics in pixels, integer pixel-centre convention: pixel (u, v) has its
  centre at (u, v). gsplat puts the centre of pixel j at j + 0.5, so the renderer is given
  ``half_pixel_K(K)``; un-projection and LiDAR projection keep the integer-centre K.
- ``ref_T_cam`` (..., 4, 4): camera -> reference, OpenCV camera axes (x right, y down, z forward), metres.
- quaternions are (w, x, y, z).
- ego frame: origin at the centre of the rear axle, x forward, y left, z up.
"""
from __future__ import annotations

import math

import torch


def half_pixel_K(K: torch.Tensor) -> torch.Tensor:
    """K for a renderer whose pixel j has its centre at j + 0.5 (gsplat): cx, cy + 0.5."""
    K = K.clone()
    K[..., 0, 2] += 0.5
    K[..., 1, 2] += 0.5
    return K


def pixel_rays(K: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """K^-1 (u, v, 1) for every pixel centre: (..., H, W, 3). The z component is 1 (not normalised), so
    a point at depth z along the optical axis is ``rays * z``."""
    v, u = torch.meshgrid(torch.arange(height, dtype=K.dtype, device=K.device),
                          torch.arange(width, dtype=K.dtype, device=K.device), indexing="ij")
    fx, fy = K[..., 0, 0, None, None], K[..., 1, 1, None, None]
    cx, cy = K[..., 0, 2, None, None], K[..., 1, 2, None, None]
    x = (u - cx) / fx
    y = (v - cy) / fy
    return torch.stack([x, y, torch.ones_like(x)], dim=-1)


def project(points_ref: torch.Tensor, ref_T_cam: torch.Tensor, K: torch.Tensor):
    """Points (..., 3) in the reference frame -> pixel coordinates (..., 2) (integer-centre convention)
    and depth z (...,) in the camera ``ref_T_cam`` (4, 4) with intrinsics ``K`` (3, 3)."""
    cam_T_ref = torch.linalg.inv(ref_T_cam)
    p = points_ref @ cam_T_ref[:3, :3].T + cam_T_ref[:3, 3]
    z = p[..., 2]
    uv = torch.stack([K[0, 0] * p[..., 0] / z + K[0, 2], K[1, 1] * p[..., 1] / z + K[1, 2]], dim=-1)
    return uv, z


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (..., 3, 3) -> unit quaternions (..., 4) as (w, x, y, z), w >= 0."""
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    # four candidates, one per largest component; pick the numerically best one
    q_abs = torch.sqrt(torch.clamp(torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                                                1 - m00 + m11 - m22, 1 - m00 - m11 + m22], dim=-1), min=0))
    cand = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1),
        torch.stack([R[..., 2, 1] - R[..., 1, 2], q_abs[..., 1] ** 2, R[..., 1, 0] + R[..., 0, 1],
                     R[..., 0, 2] + R[..., 2, 0]], dim=-1),
        torch.stack([R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] + R[..., 0, 1], q_abs[..., 2] ** 2,
                     R[..., 1, 2] + R[..., 2, 1]], dim=-1),
        torch.stack([R[..., 1, 0] - R[..., 0, 1], R[..., 2, 0] + R[..., 0, 2], R[..., 2, 1] + R[..., 1, 2],
                     q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    cand = cand / (2.0 * q_abs[..., None].clamp(min=0.1))
    best = q_abs.argmax(dim=-1)
    q = torch.gather(cand, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    q = torch.nn.functional.normalize(q, dim=-1)
    return torch.where(q[..., :1] < 0, -q, q)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a * b of quaternions (..., 4) in (w, x, y, z): the rotation b followed by a."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def chase_camera(ref_T_ego: torch.Tensor, back_m: float = 15.0, up_m: float = 9.0, pitch_down_deg: float = 18.0,
                 hfov_deg: float = 80.0, width: int = 1280, height: int = 720):
    """A virtual camera behind and above the ego vehicle, looking forward and down.

    The camera sits ``back_m`` behind and ``up_m`` above the centre of the rear axle (the ego origin) and is
    pitched down by ``pitch_down_deg``. Returns (ref_T_cam (4, 4), K (3, 3)); K is in the integer
    pixel-centre convention with the optical axis through the image centre and square pixels.
    """
    t = math.radians(pitch_down_deg)
    forward = torch.tensor([math.cos(t), 0.0, -math.sin(t)])       # camera z in the ego frame
    right = torch.tensor([0.0, -1.0, 0.0])                         # camera x (ego y points left)
    down = torch.linalg.cross(forward, right)                      # camera y
    ego_T_cam = torch.eye(4)
    ego_T_cam[:3, 0], ego_T_cam[:3, 1], ego_T_cam[:3, 2] = right, down, forward
    ego_T_cam[:3, 3] = torch.tensor([-back_m, 0.0, up_m])
    ref_T_cam = ref_T_ego.float() @ ego_T_cam.to(ref_T_ego.device)
    f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
    K = torch.tensor([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
                     device=ref_T_ego.device)
    return ref_T_cam, K
