"""Image downscaling, intrinsics bookkeeping and ego-vehicle masks.

Pixel-centre convention (used by every function here, by the LiDAR projection and by the loader):
pixel (row v, column u) covers [u - 0.5, u + 0.5) x [v - 0.5, v + 0.5), i.e. pixel centres sit at integer
coordinates. A continuous projection (u, v) falls into pixel (floor(v + 0.5), floor(u + 0.5)).
Resizing an image by sx = W'/W, sy = H'/H maps the continuous coordinate x to (x + 0.5) * s - 0.5, hence

    fx' = fx * sx,  fy' = fy * sy,  cx' = (cx + 0.5) * sx - 0.5,  cy' = (cy + 0.5) * sy - 0.5.

Renderers that put pixel centres at j + 0.5 (gsplat does) need ``intrinsics_for_half_pixel_renderer``.
Whether the Argoverse 2 calibration itself uses integer centres is not documented; we assume it does.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import FRONT_CENTER

FRONT_CENTER_MODES = ("crop", "split", "letterbox")


def scale_intrinsics(K: np.ndarray, src_wh, dst_wh) -> np.ndarray:
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    K = np.array(K, np.float64)
    out = K.copy()
    out[0, 0] = K[0, 0] * sx
    out[1, 1] = K[1, 1] * sy
    out[0, 1] = K[0, 1] * sx
    out[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    out[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return out


def crop_intrinsics(K: np.ndarray, row0: int = 0, col0: int = 0) -> np.ndarray:
    out = np.array(K, np.float64).copy()
    out[0, 2] -= col0
    out[1, 2] -= row0
    return out


def intrinsics_for_half_pixel_renderer(K: np.ndarray) -> np.ndarray:
    """K for a renderer whose pixel j has its centre at j + 0.5 (gsplat): cx, cy + 0.5."""
    out = np.array(K, np.float64).copy()
    out[..., 0, 2] += 0.5
    out[..., 1, 2] += 0.5
    return out


def project(K: np.ndarray, pts_cam: np.ndarray):
    """Camera-frame points (N, 3) -> continuous (u, v) and depth z along the optical axis."""
    z = pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (K[0, 0] * pts_cam[:, 0] + K[0, 1] * pts_cam[:, 1]) / z + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
    return u, v, z


def pixel_of(u: np.ndarray, v: np.ndarray):
    """Integer pixel indices (row, column) under the integer-centre convention."""
    return np.floor(v + 0.5).astype(np.int64), np.floor(u + 0.5).astype(np.int64)


def stored_size(cam: str, cfg: dict) -> tuple[int, int]:
    """(width, height) of the stored image of a camera."""
    key = "front_center_wh" if cam == FRONT_CENTER else "landscape_wh"
    w, h = cfg["images"][key]
    return int(w), int(h)


def stored_intrinsics(K_full: np.ndarray, full_wh, cam: str, cfg: dict) -> np.ndarray:
    return scale_intrinsics(K_full, full_wh, stored_size(cam, cfg))


def resize_to_jpeg(src_file: str, cam: str, cfg: dict) -> bytes:
    """Full decode -> INTER_AREA resize -> JPEG bytes."""
    im = cv2.imread(src_file, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(f"cannot decode {src_file}")
    small = resize_array(im, cam, cfg)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(cfg["images"]["jpeg_quality"])])
    if not ok:
        raise IOError(f"cannot encode {src_file}")
    return buf.tobytes()


def resize_array(im: np.ndarray, cam: str, cfg: dict) -> np.ndarray:
    return cv2.resize(im, stored_size(cam, cfg), interpolation=cv2.INTER_AREA)


def front_center_view(img: np.ndarray, K: np.ndarray, mode: str, cfg: dict, extra=None):
    """Turn the stored 518x684 front-center image (H, W, ...) into the training input.

    mode "crop": rows [146, 538) -> 518x392, cy -= 146. Returns (image, K).
    mode "split" (two overlapping 518x392 halves) and "letterbox" (whole image shrunk to 297x392 and padded
    to 518 wide) are reserved and not implemented yet.
    ``extra`` is an optional list of per-pixel arrays (depth, masks) cropped the same way.
    """
    if mode == "crop":
        h = cfg["images"]["landscape_wh"][1]
        r0 = int(cfg["images"]["front_center_crop_row0"])
        out = img[r0:r0 + h]
        Kc = crop_intrinsics(K, row0=r0)
        if extra is None:
            return out, Kc
        return out, Kc, [e[r0:r0 + h] for e in extra]
    if mode in ("split", "letterbox"):
        raise NotImplementedError(f"front-center mode '{mode}' is reserved but not implemented")
    raise ValueError(f"unknown front-center mode '{mode}', expected one of {FRONT_CENTER_MODES}")


def load_ego_mask(mask_file: str, cam: str, cfg: dict) -> np.ndarray:
    """Ego-body mask at the stored resolution: any coverage counts as body, then dilated."""
    m = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"cannot read ego mask {mask_file}")
    small = cv2.resize((m > 127).astype(np.float32), stored_size(cam, cfg), interpolation=cv2.INTER_AREA) > 0
    d = int(cfg["ego_mask"]["dilate_px"])
    if d > 0:
        small = cv2.dilate(small.astype(np.uint8), np.ones((2 * d + 1, 2 * d + 1), np.uint8)) > 0
    return small


def ego_masks_for_log(mask_dir: str | None, cams, cfg: dict) -> dict[str, np.ndarray]:
    """Stored-resolution ego masks for all cameras (all-False where no mask file exists)."""
    import os

    out = {}
    for cam in cams:
        w, h = stored_size(cam, cfg)
        f = os.path.join(mask_dir, f"{cam}.png") if mask_dir else None
        out[cam] = load_ego_mask(f, cam, cfg) if f and os.path.isfile(f) else np.zeros((h, w), bool)
    return out
