"""Small rigid-body and quaternion helpers, float64 throughout.

Quaternions are (qw, qx, qy, qz), scalar first, as in Argoverse 2 files.
Poses are 4x4 homogeneous matrices; ``a_T_b`` maps points from frame b to frame a.
"""
from __future__ import annotations

import numpy as np


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3). The quaternion is normalised first."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty(q.shape[:-1] + (3, 3), np.float64)
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - z * w)
    r[..., 0, 2] = 2 * (x * z + y * w)
    r[..., 1, 0] = 2 * (x * y + z * w)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - x * w)
    r[..., 2, 0] = 2 * (x * z - y * w)
    r[..., 2, 1] = 2 * (y * z + x * w)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def rotmat_to_quat(r: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) wxyz with qw >= 0."""
    from scipy.spatial.transform import Rotation

    r = np.asarray(r, np.float64)
    flat = r.reshape(-1, 3, 3)
    q = Rotation.from_matrix(flat).as_quat(scalar_first=True)
    q = np.where(q[:, :1] < 0, -q, q)
    return q.reshape(r.shape[:-2] + (4,))


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation of wxyz quaternions, vectorised over the leading axis.

    ``alpha`` may lie outside [0, 1]; the rotation is then extrapolated along the same great circle.
    """
    q0 = np.asarray(q0, np.float64)
    q1 = np.asarray(q1, np.float64)
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    alpha = np.asarray(alpha, np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_t = np.sin(theta)
    small = sin_t < 1e-8
    safe = np.where(small, 1.0, sin_t)
    w0 = np.where(small, 1.0 - alpha, np.sin((1.0 - alpha) * theta) / safe)
    w1 = np.where(small, alpha, np.sin(alpha * theta) / safe)
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def make_se3(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """(..., 3, 3), (..., 3) -> (..., 4, 4)."""
    rot = np.asarray(rot, np.float64)
    trans = np.asarray(trans, np.float64)
    t = np.zeros(rot.shape[:-2] + (4, 4), np.float64)
    t[..., :3, :3] = rot
    t[..., :3, 3] = trans
    t[..., 3, 3] = 1.0
    return t


def se3_from_quat_trans(q: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return make_se3(quat_to_rotmat(q), trans)


def se3_inv(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, np.float64)
    rot = t[..., :3, :3]
    rt = np.swapaxes(rot, -1, -2)
    return make_se3(rt, -(rt @ t[..., :3, 3:4])[..., 0])


def transform_points(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply one 4x4 transform to (N, 3) points."""
    t = np.asarray(t, np.float64)
    return pts @ t[:3, :3].T + t[:3, 3]
