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


class FrontCenterView:
    """One training view cut from the stored 518x684 front-center image; geometry follows the same rules as
    ``scale_intrinsics`` / ``crop_intrinsics`` (integer pixel centres).

    The view takes stored rows [row0, row0 + rows), scales them by (sx, sy) and pads ``col_pad`` columns on
    the left, giving an image of ``out_wh``. Images are resized with INTER_AREA, masks with nearest
    neighbour, sparse depth by moving every point to its new pixel (nearest point kept, flags follow it).
    """

    def __init__(self, row0: int, rows: int, out_wh, scaled_wh=None, col_pad: int = 0, name: str = ""):
        self.row0, self.rows = int(row0), int(rows)
        self.out_wh = (int(out_wh[0]), int(out_wh[1]))
        self.scaled_wh = (int(scaled_wh[0]), int(scaled_wh[1])) if scaled_wh is not None else None
        self.col_pad = int(col_pad)
        self.name = name

    def _src_wh(self, width: int):
        return (int(width), self.rows)

    def K(self, K_stored: np.ndarray, width: int) -> np.ndarray:
        Kc = crop_intrinsics(K_stored, row0=self.row0)
        if self.scaled_wh is not None:
            Kc = scale_intrinsics(Kc, self._src_wh(width), self.scaled_wh)
        return crop_intrinsics(Kc, col0=-self.col_pad)

    def _place(self, small: np.ndarray, fill) -> np.ndarray:
        W, H = self.out_wh
        if small.shape[1] == W and small.shape[0] == H:
            return small
        out = np.full((H, W) + small.shape[2:], fill, small.dtype)
        out[:, self.col_pad:self.col_pad + small.shape[1]] = small
        return out

    def image(self, img: np.ndarray) -> np.ndarray:
        band = img[self.row0:self.row0 + self.rows]
        if self.scaled_wh is not None:
            band = cv2.resize(band, self.scaled_wh, interpolation=cv2.INTER_AREA)
        return self._place(band, 0)

    def mask(self, m: np.ndarray, pad_value=0) -> np.ndarray:
        band = m[self.row0:self.row0 + self.rows]
        if self.scaled_wh is not None:
            dt = band.dtype
            band = cv2.resize(band.astype(np.uint8), self.scaled_wh, interpolation=cv2.INTER_NEAREST).astype(dt)
        return self._place(band, pad_value)

    def padding(self) -> np.ndarray:
        """True on the padded columns (no image content)."""
        W, H = self.out_wh
        p = np.ones((H, W), bool)
        w = self.scaled_wh[0] if self.scaled_wh is not None else W
        p[:, self.col_pad:self.col_pad + w] = False
        return p

    def sparse(self, vu: np.ndarray, z: np.ndarray, flag: np.ndarray, width: int):
        """Move sparse points (v, u at the stored resolution) into this view; nearest point per pixel."""
        v = vu[:, 0].astype(np.float64) - self.row0
        u = vu[:, 1].astype(np.float64)
        if self.scaled_wh is not None:
            sx = self.scaled_wh[0] / width
            sy = self.scaled_wh[1] / self.rows
            u = (u + 0.5) * sx - 0.5
            v = (v + 0.5) * sy - 0.5
        u = u + self.col_pad
        vi, ui = pixel_of(u, v)
        W, H = self.out_wh
        ok = (vi >= 0) & (vi < H) & (ui >= 0) & (ui < W)
        vi, ui, zz, ff = vi[ok], ui[ok], np.asarray(z)[ok], np.asarray(flag)[ok]
        if len(zz) and self.scaled_wh is not None:
            pix = vi * W + ui
            order = np.lexsort((zz.astype(np.float32), pix))  # by pixel, nearest first
            keep = order[np.r_[True, pix[order][1:] != pix[order][:-1]]]
            vi, ui, zz, ff = vi[keep], ui[keep], zz[keep], ff[keep]
        return np.stack([vi, ui], 1).astype(np.uint16), zz, ff


def front_center_views(mode: str, cfg: dict) -> list[FrontCenterView]:
    """Training views of the stored front-center image for each ``front_center_mode``.

    - "crop": rows [146, 538) -> one 518x392 view (full-resolution rows ~437..1610), cy -= 146.
    - "split": rows [0, 392) and [292, 684) -> two 518x392 views overlapping by 100 rows, cy -= 0 / 292.
    - "letterbox": the whole 518x684 image shrunk to 297x392 and padded with 110 / 111 black columns
      to 518x392 (a second resize of the stored image).
    """
    W, H = (int(x) for x in cfg["images"]["landscape_wh"])
    Wf, Hf = (int(x) for x in cfg["images"]["front_center_wh"])
    if mode == "crop":
        return [FrontCenterView(int(cfg["images"]["front_center_crop_row0"]), H, (W, H), name="crop")]
    if mode == "split":
        return [FrontCenterView(0, H, (W, H), name="top"), FrontCenterView(Hf - H, H, (W, H), name="bottom")]
    if mode == "letterbox":
        w = int(round(Wf * H / Hf))
        return [FrontCenterView(0, Hf, (W, H), scaled_wh=(w, H), col_pad=(W - w) // 2, name="letterbox")]
    raise ValueError(f"unknown front-center mode '{mode}', expected one of {FRONT_CENTER_MODES}")


def front_center_view(img: np.ndarray, K: np.ndarray, mode: str, cfg: dict, extra=None):
    """Single-view helper for "crop" (used by the overlays): returns (image, K[, cropped per-pixel arrays])."""
    if mode != "crop":
        if mode in FRONT_CENTER_MODES:
            raise ValueError("front_center_view handles 'crop' only; use front_center_views")
        raise ValueError(f"unknown front-center mode '{mode}', expected one of {FRONT_CENTER_MODES}")
    view = front_center_views("crop", cfg)[0]
    out, Kc = view.image(img), view.K(K, img.shape[1])
    if extra is None:
        return out, Kc
    return out, Kc, [view.mask(e) for e in extra]


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

    if mask_dir and not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"ego mask directory not found: {mask_dir}")
    out = {}
    for cam in cams:
        w, h = stored_size(cam, cfg)
        f = os.path.join(mask_dir, f"{cam}.png") if mask_dir else None
        out[cam] = load_ego_mask(f, cam, cfg) if f and os.path.isfile(f) else np.zeros((h, w), bool)
    return out
