import json
import os
import tempfile
import unittest

import _util  # noqa: F401
import cv2
import numpy as np
import pandas as pd
import torch

from realityview.data.config import CAMS, FRONT_CENTER, default_config, init_root_config
from realityview.data.dataset import Stage1Dataset, valid_moments
from realityview.data.geometry import make_se3, quat_to_rotmat
from realityview.data.images import scale_intrinsics, stored_size
from realityview.data.lidar_depth import SparseDepth
from realityview.data.store import depth_relpath, depth_to_bytes, img_relpath, pack_log, write_bytes

N_FRAMES = 30
T0 = 7_000_000_000


def build_fake_root(root, with_zip=False):
    """A tiny processed root with one log in the on-disk format written by preprocess.py."""
    cfg = default_config()
    init_root_config(root, cfg)
    os.makedirs(os.path.join(root, "ego_masks"))
    for cam in CAMS:
        w, h = stored_size(cam, cfg)
        m = np.zeros((h, w), np.uint8)
        if cam == FRONT_CENTER:
            m[600:] = 255
        cv2.imwrite(os.path.join(root, "ego_masks", f"{cam}.png"), m)
    log = os.path.join(root, "logs", "val", "fake")
    rng = np.random.default_rng(0)
    rows, mrows = [], []
    K_full = np.array([[1700.0, 0, 1024.0], [0, 1700.0, 775.0], [0, 0, 1]])
    meta = {"K_stored": {}, "ego_T_cam": {}, "log_id": "fake", "split": "val", "city": "XXX"}
    for c_i, cam in enumerate(CAMS):
        w, h = stored_size(cam, cfg)
        full = (1550, 2048) if cam == FRONT_CENTER else (2048, 1550)
        K = K_full if cam != FRONT_CENTER else np.array([[1780.0, 0, 775.0], [0, 1780.0, 1023.0], [0, 0, 1]])
        meta["K_stored"][cam] = scale_intrinsics(K, full, (w, h)).tolist()
        yaw = c_i * 2 * np.pi / 7
        R_yaw = quat_to_rotmat(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        R_cam = R_yaw @ np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
        meta["ego_T_cam"][cam] = make_se3(R_cam, [1.5, 0.0, 1.4]).tolist()
        for f in range(N_FRAMES):
            ts = T0 + f * 50_000_000 + c_i * 1_000_000
            rows.append({"cam": cam, "frame": f, "timestamp_ns": ts, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
                         "tx_m": 1000.0 + 0.4 * f, "ty_m": 50.0, "tz_m": 10.0, "k0": f // 2})
            img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            img[0, 0] = (0, 0, 255)  # BGR red marker
            write_bytes(os.path.join(log, img_relpath(cam, ts)), cv2.imencode(".png", img)[1].tobytes())
            n = 300
            pix = np.sort(rng.choice(h * w, n, replace=False))
            d = SparseDepth(np.stack([pix // w, pix % w], 1).astype(np.uint16),
                            rng.uniform(0.5, 250, n).astype(np.float16), rng.integers(0, 8, n).astype(np.uint8))
            for layer in ("acc", "single"):
                write_bytes(os.path.join(log, depth_relpath(layer, cam, ts)), depth_to_bytes(d))
    for f in range(N_FRAMES):
        r = {"i": f, "complete": f != 3}
        for c_i, cam in enumerate(CAMS):
            r[f"idx_{cam}"] = f
            r[f"ts_{cam}"] = T0 + f * 50_000_000 + c_i * 1_000_000
        mrows.append(r)
    pd.DataFrame(rows).to_parquet(os.path.join(log, "frames.parquet"))
    pd.DataFrame(mrows).to_parquet(os.path.join(log, "moments.parquet"))
    pd.DataFrame({"cam": [], "frame": []}).to_parquet(os.path.join(log, "boxes.parquet"))
    with open(os.path.join(log, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(log, "DONE"), "w") as f:
        json.dump({"checks_passed": True}, f)
    if with_zip:
        pack_log(log, log + ".zip")
        import shutil

        shutil.rmtree(log)
    return cfg


class TestValidMoments(unittest.TestCase):
    def test_offsets(self):
        m = pd.DataFrame({"complete": [True] * 20})
        m.loc[5, "complete"] = False
        v = valid_moments(m, (0, -10))
        self.assertEqual(v.tolist(), [i for i in range(10, 20) if i != 15])
        v2 = valid_moments(m, (0, -5, -10))
        self.assertEqual(v2.tolist(), [i for i in range(11, 20) if i != 15])


class TestStage1Synthetic(unittest.TestCase):
    def check_sample(self, s, n_imgs):
        self.assertEqual(tuple(s["images"].shape), (n_imgs, 3, 392, 518))
        self.assertEqual(s["images"].dtype, torch.uint8)
        self.assertEqual(tuple(s["K"].shape), (n_imgs, 3, 3))
        self.assertEqual(tuple(s["ref_T_cam"].shape), (n_imgs, 4, 4))
        self.assertEqual(tuple(s["depth"].shape), (n_imgs, 392, 518))
        self.assertEqual(s["depth"].dtype, torch.float32)
        self.assertEqual(s["ego_mask"].dtype, torch.bool)
        d = s["depth"]
        pos = d[d > 0]
        self.assertTrue(((pos >= 0.49) & (pos <= 250.2)).all())
        self.assertTrue(torch.allclose(s["ref_T_cam"][0], torch.eye(4), atol=1e-5))
        self.assertEqual(float(s["dt"][0]), 0.0)
        R = s["ref_T_cam"][:, :3, :3].double()
        eye = torch.eye(3, dtype=torch.float64).expand_as(R)
        self.assertTrue(torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-5))

    def test_default(self):
        for with_zip in (False, True):
            with tempfile.TemporaryDirectory() as root:
                build_fake_root(root, with_zip=with_zip)
                ds = Stage1Dataset(root, "val", ["fake"])
                # moments 10..29 except 3+10=13 (t-10 incomplete) and 3 itself (< 10)
                self.assertEqual(len(ds), len([i for i in range(10, N_FRAMES) if i != 13]))
                s = ds[0]
                self.check_sample(s, 14)
                self.assertEqual(s["moment"], 10)
                self.assertEqual(s["cam_id"].tolist(), list(range(7)) * 2)
                self.assertEqual(s["slot"].tolist(), [0] * 7 + [1] * 7)
                self.assertAlmostEqual(float(s["dt"][7]), -0.5, places=6)
                # RGB order: the BGR red marker at (0, 0) of landscape images is red in channel 0
                self.assertGreater(int(s["images"][1, 0, 0, 0]), 200)
                # front-center: K is cropped by 146 rows, ego mask rows 600.. become rows 454.. (outside crop)
                Kfc = np.array(ds._log(0)["meta"]["K_stored"][FRONT_CENTER])
                self.assertAlmostEqual(float(s["K"][0, 1, 2]), Kfc[1, 2] - 146, places=3)
                self.assertFalse(s["ego_mask"][0].any())
                # t-0.5 front-center camera is 0.5 s * 8 m/s = 4 m behind along x
                self.assertAlmostEqual(float(s["ref_T_cam"][7, :3, 3].norm()), 4.0, places=3)

    def test_switches(self):
        with tempfile.TemporaryDirectory() as root:
            build_fake_root(root)
            ds = Stage1Dataset(root, "val", ["fake"], input_offsets=(0, -5, -10), load_single_layer=True,
                               with_box_masks=True)
            s = ds[0]
            self.check_sample(s, 21)
            self.assertIn("depth_single", s)
            self.assertEqual(tuple(s["box_mask"].shape), (21, 392, 518))
            for mode, n_img in (("split", 16), ("letterbox", 14)):
                s = Stage1Dataset(root, "val", ["fake"], front_center_mode=mode)[0]
                self.check_sample(s, n_img)
            with self.assertRaises(ValueError):
                Stage1Dataset(root, "val", ["fake"], front_center_mode="other")


class TestReferenceAndViews(unittest.TestCase):
    def test_reference_is_front_center_at_t(self):
        """With inputs that exclude t, poses and dt still refer to the front-center camera at t."""
        with tempfile.TemporaryDirectory() as root:
            build_fake_root(root)
            base = Stage1Dataset(root, "val", ["fake"])
            ds = Stage1Dataset(root, "val", ["fake"], input_offsets=(-10,))
            # t must itself be complete: moment 3 is incomplete, so t = 3 is never a sample
            self.assertNotIn(3, [i for _, i in ds.samples])
            k = [n for n, (_, i) in enumerate(ds.samples) if i == 20][0]
            kb = [n for n, (_, i) in enumerate(base.samples) if i == 20][0]
            s, sb = ds[k], base[kb]
            self.assertEqual(tuple(s["images"].shape), (7, 3, 392, 518))
            # the t-10 slot of the default sample equals the only slot here
            self.assertTrue(torch.allclose(s["ref_T_cam"], sb["ref_T_cam"][7:], atol=1e-5))
            self.assertTrue(torch.allclose(s["dt"], sb["dt"][7:]))
            self.assertAlmostEqual(float(s["dt"][0]), -0.5, places=6)

    def test_front_center_views_in_sample(self):
        with tempfile.TemporaryDirectory() as root:
            build_fake_root(root)
            Kfc = None
            for mode, n_fc in (("crop", 1), ("split", 2), ("letterbox", 1)):
                ds = Stage1Dataset(root, "val", ["fake"], front_center_mode=mode, with_box_masks=True)
                s = ds[0]
                Kfc = np.array(ds._log(0)["meta"]["K_stored"][FRONT_CENTER])
                fc = (s["cam_id"] == 0) & (s["slot"] == 0)
                self.assertEqual(int(fc.sum()), n_fc)
                self.assertEqual(s["view"][fc].tolist(), list(range(n_fc)))
                if mode == "split":
                    self.assertAlmostEqual(float(s["K"][1, 1, 2]), Kfc[1, 2] - 292, places=3)
                    # the ego mask (stored rows 600..683) appears in the lower view only
                    self.assertFalse(s["ego_mask"][0].any())
                    self.assertTrue(s["ego_mask"][1][600 - 292:].all())
                if mode == "letterbox":
                    self.assertTrue(s["ego_mask"][0][:, :110].all())  # padding is never supervised
                    self.assertAlmostEqual(float(s["K"][0, 0, 2]), (Kfc[0, 2] + 0.5) * 297 / 518 - 0.5 + 110,
                                           places=3)
                    self.assertFalse(s["images"][0][:, :, :110].any())
                # depth stays within range and on the ego mask's complement where the source was
                pos = s["depth"][s["depth"] > 0]
                self.assertTrue(((pos >= 0.49) & (pos <= 250.2)).all())

    def test_missing_ego_masks_fail(self):
        with tempfile.TemporaryDirectory() as root:
            build_fake_root(root)
            os.remove(os.path.join(root, "ego_masks", "ring_side_left.png"))
            with self.assertRaises(FileNotFoundError):
                Stage1Dataset(root, "val", ["fake"])

    def test_overlapping_root_lists_fail(self):
        with tempfile.TemporaryDirectory() as root:
            build_fake_root(root)
            os.makedirs(os.path.join(root, "splits"))
            for name, ids in (("train.txt", ["a", "b"]), ("dev.txt", ["c"]), ("val.txt", ["b"])):
                with open(os.path.join(root, "splits", name), "w") as f:
                    f.write("\n".join(ids) + "\n")
            with self.assertRaises(ValueError):
                Stage1Dataset(root, "val", ["fake"])


class TestStage1Real(unittest.TestCase):
    def test_real(self):
        root = _util.processed_root()
        split = os.environ.get("RV_TEST_SPLIT", "val")
        log_id = os.environ.get("RV_TEST_LOG")
        if not log_id:
            with open(os.path.join(_util.REPO, "data_lists", "av2_val_report20.txt")) as f:
                log_id = f.readline().strip()
        ds = Stage1Dataset(root, split, [log_id], allow_missing_logs=True)
        if len(ds) == 0:
            raise unittest.SkipTest(f"{split}/{log_id} not processed under RV_PROCESSED_ROOT")
        self.assertGreaterEqual(len(ds), 300)
        s = ds[len(ds) // 2]
        self.assertEqual(tuple(s["images"].shape), (14, 3, 392, 518))
        self.assertTrue(torch.allclose(s["ref_T_cam"][0], torch.eye(4), atol=1e-5))
        self.assertLess(float(s["dt"][:7].abs().max()), 0.0230)
        self.assertLess(float((s["dt"][7:] + 0.5).abs().max()), 0.0230)
        cov = (s["depth"] > 0).float().mean().item()
        self.assertGreater(cov, 0.05)
        pos = s["depth"][s["depth"] > 0]
        self.assertTrue(((pos >= 0.49) & (pos <= 250.2)).all())
        # cameras sit within ~2 m of the front-center camera at t
        self.assertLess(float(s["ref_T_cam"][:7, :3, 3].norm(dim=1).max()), 2.5)


if __name__ == "__main__":
    unittest.main()
