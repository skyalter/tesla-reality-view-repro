"""On-disk format of processed logs: deterministic npz depth files, loose directories and stored zips.

Per log (loose directory ``logs/<split>/<log_id>/`` while developing, or one uncompressed zip
``logs/<split>/<log_id>.zip`` after checks; both are read through ``LogStore``):

    meta.json                  log id, city, sizes, stored intrinsics, extrinsics, config + sha256, code commit
    frames.parquet             one row per camera image: camera, frame, timestamp, exact ego pose, k0, stats
    moments.parquet            one row per front-center frame: frames of all cameras, validity, displacement
    sweeps.parquet             one row per LiDAR sweep: timestamp, ego pose, point count
    boxes.parquet              interpolated boxes for every camera image (city frame)
    img/<cam>/<ts>.jpg         downscaled images
    depth/acc/<cam>/<ts>.npz   accumulated sparse depth: v, u uint16 (N,), z float16 (N,), flag uint8 (N,)
    depth/single/<cam>/<ts>.npz  single-sweep sparse depth, same keys
    sky/<cam>/<ts>.png         raw sky score x 100 as uint8, at the stored image size (scripts/sky_scores.py)
    sky/DONE                   written when every image of the log has its sky score
    CHECKS.json, DONE
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile

import numpy as np

from .lidar_depth import SparseDepth

_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def npz_bytes(arrays: dict[str, np.ndarray], compress: bool = True) -> bytes:
    """npz with fixed member timestamps, so identical arrays give identical bytes."""
    buf = io.BytesIO()
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buf, "w", compression=method) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(name + ".npy", date_time=_FIXED_TIME)
            info.compress_type = method
            b = io.BytesIO()
            np.lib.format.write_array(b, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            zf.writestr(info, b.getvalue())
    return buf.getvalue()


def depth_to_bytes(d: SparseDepth) -> bytes:
    """Rows and columns are stored as two separate uint16 arrays (sorted by pixel), which deflates ~40%
    better than an interleaved (N, 2) array."""
    vu = d.vu.astype(np.uint16)
    return npz_bytes({"v": np.ascontiguousarray(vu[:, 0]), "u": np.ascontiguousarray(vu[:, 1]),
                      "z": d.z.astype(np.float16), "flag": d.flag.astype(np.uint8)})


def depth_from_bytes(b: bytes) -> SparseDepth:
    with np.load(io.BytesIO(b), allow_pickle=False) as f:
        return SparseDepth(np.stack([f["v"], f["u"]], 1), f["z"], f["flag"])


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_bytes(path: str, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def img_relpath(cam: str, ts: int) -> str:
    return f"img/{cam}/{int(ts)}.jpg"


def depth_relpath(layer: str, cam: str, ts: int) -> str:
    assert layer in ("acc", "single")
    return f"depth/{layer}/{cam}/{int(ts)}.npz"


SKY_DONE = "sky/DONE"


def sky_relpath(cam: str, ts: int) -> str:
    return f"sky/{cam}/{int(ts)}.png"


class LogStore:
    """Read files of one processed log from a loose directory or a stored zip."""

    def __init__(self, path: str):
        self.path = path
        self.is_zip = path.endswith(".zip")
        self._zf = None
        self._pid = None

    def _zip(self):
        if self._zf is None or self._pid != os.getpid():  # one handle per process (DataLoader workers)
            self._zf = zipfile.ZipFile(self.path, "r")
            self._pid = os.getpid()
        return self._zf

    def read(self, rel: str) -> bytes:
        if self.is_zip:
            return self._zip().read(rel)
        with open(os.path.join(self.path, rel), "rb") as f:
            return f.read()

    def exists(self, rel: str) -> bool:
        if self.is_zip:
            try:
                self._zip().getinfo(rel)
                return True
            except KeyError:
                return False
        return os.path.exists(os.path.join(self.path, rel))

    def json(self, rel: str):
        return json.loads(self.read(rel))

    def parquet(self, rel: str):
        import pandas as pd

        return pd.read_parquet(io.BytesIO(self.read(rel)))

    def depth(self, layer: str, cam: str, ts: int) -> SparseDepth:
        return depth_from_bytes(self.read(depth_relpath(layer, cam, ts)))

    def sky(self, cam: str, ts: int) -> np.ndarray:
        """Stored sky score of one image: uint8 (H, W), raw score x 100 (see realityview/models/sky.py)."""
        import cv2

        m = cv2.imdecode(np.frombuffer(self.read(sky_relpath(cam, ts)), np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise IOError(f"cannot decode {self.path}/{sky_relpath(cam, ts)}")
        return m

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_zf"] = None
        d["_pid"] = None
        return d


def pack_log(log_dir: str, zip_path: str) -> int:
    """Pack a loose processed log directory into one uncompressed zip (sorted members, fixed times)."""
    files = []
    for root, _, names in os.walk(log_dir):
        for n in names:
            full = os.path.join(root, n)
            files.append(os.path.relpath(full, log_dir).replace(os.sep, "/"))
    files.sort()
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for rel in files:
            info = zipfile.ZipInfo(rel, date_time=_FIXED_TIME)
            info.compress_type = zipfile.ZIP_STORED
            with open(os.path.join(log_dir, rel), "rb") as f:
                zf.writestr(info, f.read())
    os.replace(tmp, zip_path)
    return len(files)


def find_log(processed_root: str, split: str, log_id: str) -> str | None:
    """Path of a processed log (directory preferred over zip), or None."""
    d = os.path.join(processed_root, "logs", split, log_id)
    if os.path.isdir(d):
        return d
    z = d + ".zip"
    return z if os.path.isfile(z) else None
