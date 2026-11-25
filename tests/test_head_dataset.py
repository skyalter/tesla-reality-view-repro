import json
import os
import tempfile
import time
import unittest

import _util  # noqa: F401
import cv2
import numpy as np
import pandas as pd
import torch

from realityview.data.config import CAMS, FRONT_CENTER, default_config, init_root_config
from realityview.data.dataset import Stage1Dataset
from realityview.data.geometry import make_se3, quat_to_rotmat, se3_inv
from realityview.data.head_dataset import Stage1HeadDataset, target_candidates
from realityview.data.images import scale_intrinsics, stored_size
from realityview.data.lidar_depth import SparseDepth
from realityview.data.store import SKY_DONE, depth_relpath, depth_to_bytes, img_relpath, sky_relpath, write_bytes

N_FRAMES = 60  # valid_stage2: 20 <= i <= 39
T0 = 7_000_000_000
INCOMPLETE = (25, 54)
STANDING = (30, 31, 32)  # disp_win_max_m below 0.5 m


def cam_ts(c_i, f):
    return T0 + f * 50_000_000 + c_i * 1_000_000


def city_T_ego_at(ts):
    """Exact ego pose of the fake log at a timestamp: a curve with turning heading."""
    tau = (ts - T0) / 1e9
    yaw = 0.3 * tau
    R = quat_to_rotmat(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
    return make_se3(R, [1000.0 + 8.0 * tau, 50.0 + 2.0 * tau ** 2, 10.0])


def ego_T_cam_of(c_i):
    yaw = c_i * 2 * np.pi / 7
    R_yaw = quat_to_rotmat(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
    R_cam = R_yaw @ np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
    return make_se3(R_cam, [1.5, 0.1 * c_i, 1.4])


def box_rows(cam, f, ts):
    """A moving car 10 m ahead and a pedestrian 8 m ahead-left of the ego, interpolated to (cam, frame)."""
    T = city_T_ego_at(ts)
    q = np.array([np.cos(0.15 * (ts - T0) / 1e9), 0, 0, np.sin(0.15 * (ts - T0) / 1e9)])
    out = []
    for local, size, moving, person in (((10.0, 0.0, 0.8), (4.5, 1.9, 1.6), True, False),
                                        ((8.0, 3.0, 0.9), (0.6, 0.6, 1.8), False, True)):
        c = T[:3, :3] @ np.array(local) + T[:3, 3]
        out.append({"cam": cam, "frame": f, "cx": c[0], "cy": c[1], "cz": c[2], "cqw": q[0], "cqx": q[1],
                    "cqy": q[2], "cqz": q[3], "length_m": size[0], "width_m": size[1], "height_m": size[2],
                    "vx": 8.0 if moving else 0.0, "vy": 0.0, "vz": 0.0, "moving": moving, "person_like": person})
    return out


def build_head_root(root):
    """A processed root with one log "fake" of N_FRAMES moments, boxes and sky scores."""
    cfg = default_config()
    init_root_config(root, cfg)
    os.makedirs(os.path.join(root, "ego_masks"))
    for cam in CAMS:
        w, h = stored_size(cam, cfg)
        m = np.zeros((h, w), np.uint8)
        m[600 if cam == FRONT_CENTER else 360:] = 255
        cv2.imwrite(os.path.join(root, "ego_masks", f"{cam}.png"), m)
    log = os.path.join(root, "logs", "val", "fake")
    rng = np.random.default_rng(0)
    rows, mrows, boxes = [], [], []
    K_full = np.array([[1700.0, 0, 1024.0], [0, 1700.0, 775.0], [0, 0, 1]])
    meta = {"K_stored": {}, "ego_T_cam": {}, "log_id": "fake", "split": "val", "city": "XXX"}
    for c_i, cam in enumerate(CAMS):
        w, h = stored_size(cam, cfg)
        full = (1550, 2048) if cam == FRONT_CENTER else (2048, 1550)
        K = K_full if cam != FRONT_CENTER else np.array([[1780.0, 0, 775.0], [0, 1780.0, 1023.0], [0, 0, 1]])
        meta["K_stored"][cam] = scale_intrinsics(K, full, (w, h)).tolist()
        meta["ego_T_cam"][cam] = ego_T_cam_of(c_i).tolist()
        for f in range(N_FRAMES):
            ts = cam_ts(c_i, f)
            T = city_T_ego_at(ts)
            q = np.array([np.cos(0.3 * (ts - T0) / 1e9 / 2), 0, 0, np.sin(0.3 * (ts - T0) / 1e9 / 2)])
            rows.append({"cam": cam, "frame": f, "timestamp_ns": ts, "qw": q[0], "qx": q[1], "qy": q[2],
                         "qz": q[3], "tx_m": T[0, 3], "ty_m": T[1, 3], "tz_m": T[2, 3], "k0": f // 2})
            small = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            write_bytes(os.path.join(log, img_relpath(cam, ts)), cv2.imencode(".png", img)[1].tobytes())
            n = 300
            pix = np.sort(rng.choice(h * w, n, replace=False))
            d = SparseDepth(np.stack([pix // w, pix % w], 1).astype(np.uint16),
                            rng.uniform(0.5, 250, n).astype(np.float16), rng.integers(0, 8, n).astype(np.uint8))
            for layer in ("acc", "single"):
                write_bytes(os.path.join(log, depth_relpath(layer, cam, ts)), depth_to_bytes(d))
            sky = np.zeros((h, w), np.uint8)
            sky[:40 + f] = 120
            write_bytes(os.path.join(log, sky_relpath(cam, ts)), cv2.imencode(".png", sky)[1].tobytes())
            boxes.extend(box_rows(cam, f, ts))
    complete = np.array([f not in INCOMPLETE for f in range(N_FRAMES)])
    for f in range(N_FRAMES):
        v1 = f >= 10 and complete[f] and complete[f - 10]
        r = {"i": f, "complete": bool(complete[f]), "valid_stage1": bool(v1),
             "valid_stage2": bool(v1 and 20 <= f <= N_FRAMES - 21),
             "disp_win_max_m": 0.1 if f in STANDING else 3.0}
        for c_i, cam in enumerate(CAMS):
            r[f"idx_{cam}"] = f
            r[f"ts_{cam}"] = cam_ts(c_i, f)
        mrows.append(r)
    pd.DataFrame(rows).to_parquet(os.path.join(log, "frames.parquet"))
    pd.DataFrame(mrows).to_parquet(os.path.join(log, "moments.parquet"))
    pd.DataFrame(boxes).to_parquet(os.path.join(log, "boxes.parquet"))
    with open(os.path.join(log, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(log, SKY_DONE), "w") as f:
        json.dump({"images": len(rows)}, f)
    with open(os.path.join(log, "DONE"), "w") as f:
        json.dump({"checks_passed": True}, f)
    return complete


# moments that pass valid_stage2, the default input offsets and the displacement threshold
MOMENTS = [i for i in range(20, 40) if i not in INCOMPLETE and i - 10 not in INCOMPLETE and i not in STANDING]

PER_VIEW = ("images", "K", "ref_T_cam", "dt", "cam_id", "view", "timestamp_ns", "depth", "depth_flag",
            "ego_mask", "box_mask", "sky")


class TestHeadDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = cls._tmp.name
        cls.complete = build_head_root(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def ds(self, **kw):
        return Stage1HeadDataset(self.root, "val", ["fake"], **kw)

    def test_random_targets(self):
        ds = self.ds()
        self.assertEqual([i for _, i in ds.samples], MOMENTS)
        self.assertEqual(len(ds.sample_offsets), len(ds))
        for (_, i), k in zip(ds.samples, ds.sample_offsets):
            self.assertLessEqual(abs(k), 20)
            for o in (0, -10):
                self.assertGreater(abs(k - o), 3)
            self.assertTrue(self.complete[i + k])
            self.assertIn(k, target_candidates(self.complete, i, (0, -10), 20, 3))
        # 27 candidates when nothing nearby is incomplete; incomplete moments are never targets
        self.assertEqual(len(target_candidates(np.ones(60, bool), 30, (0, -10), 20, 3)), 27)
        self.assertNotIn(20, target_candidates(self.complete, 34, (0, -10), 20, 3))
        self.assertNotIn(5, target_candidates(self.complete, 20, (0, -10), 20, 3))
        # deterministic per index and independent of construction / reading order
        again = self.ds()
        self.assertEqual(again.sample_offsets, ds.sample_offsets)
        idx = len(ds) // 2
        _ = again[0]
        self.assertEqual(int(again[idx]["tgt_offset"]), ds.sample_offsets[idx])
        self.assertTrue(torch.equal(again[idx]["tgt_images"], ds[idx]["tgt_images"]))
        self.assertGreater(len(set(ds.sample_offsets)), 3)
        other = self.ds(seed=1)
        self.assertEqual(other.samples, ds.samples)
        self.assertNotEqual(other.sample_offsets, ds.sample_offsets)
        # the window and the gap are configurable
        narrow = self.ds(max_target_frames=6, min_gap_frames=2)
        for (_, i), k in zip(narrow.samples, narrow.sample_offsets):
            self.assertIn(k, (-6, -5, -4, -3, 3, 4, 5, 6))

    def test_fixed_offsets(self):
        ds = self.ds(target_offsets=[0, 10, 20])
        expect = [(i, k) for i in MOMENTS for k in (0, 10, 20) if self.complete[i + k]]
        self.assertEqual(len(expect), 3 * len(MOMENTS) - 1)  # 34 + 20 = 54 is incomplete
        self.assertEqual([(i, k) for (_, i), k in zip(ds.samples, ds.sample_offsets)], expect)
        n = expect.index((MOMENTS[3], 10))
        s = ds[n]
        self.assertEqual(int(s["tgt_offset"]), 10)
        self.assertEqual(s["tgt_offset"].dtype, torch.int64)
        self.assertEqual(s["moment"], MOMENTS[3])
        # offset 0: the target is input slot 0
        s0 = ds[expect.index((MOMENTS[3], 0))]
        for key in PER_VIEW:
            self.assertTrue(torch.equal(s0["tgt_" + key], s0[key][:7]), key)
        with self.assertRaises(ValueError):
            self.ds(target_offsets="all")

    def test_target_pose_and_ego_frame(self):
        ds = self.ds(target_offsets=[-17, 13])
        n = [m for m, ((_, i), k) in enumerate(zip(ds.samples, ds.sample_offsets)) if i == 27 and k == 13][0]
        s = ds[n]
        cam_T_ref = se3_inv(city_T_ego_at(cam_ts(0, 27)) @ ego_T_cam_of(0))
        for c_i in range(7):
            want = cam_T_ref @ city_T_ego_at(cam_ts(c_i, 40)) @ ego_T_cam_of(c_i)
            np.testing.assert_allclose(s["tgt_ref_T_cam"][c_i].numpy(), want, atol=2e-5)
            self.assertAlmostEqual(float(s["tgt_dt"][c_i]), 0.65 + c_i * 1e-3, places=6)
            self.assertEqual(int(s["tgt_timestamp_ns"][c_i]), cam_ts(c_i, 40))
        # the inputs use the same reference
        np.testing.assert_allclose(s["ref_T_cam"][8].numpy(),
                                   cam_T_ref @ city_T_ego_at(cam_ts(1, 17)) @ ego_T_cam_of(1), atol=2e-5)
        # reference = front-center camera at t, hence ref_T_ego = inverse of its extrinsics
        np.testing.assert_allclose(s["ref_T_ego"].numpy(), se3_inv(ego_T_cam_of(0)), atol=1e-6)
        self.assertEqual(s["ref_T_ego"].dtype, torch.float32)

    def test_min_displacement(self):
        moving = self.ds()
        allm = self.ds(min_displacement_m=0)
        self.assertEqual(sorted(set(i for _, i in allm.samples)), sorted(MOMENTS + list(STANDING)))
        for i in STANDING:
            self.assertNotIn(i, [i2 for _, i2 in moving.samples])
        self.assertEqual(len(self.ds(min_displacement_m=5.0)), 0)
        self.assertGreaterEqual(float(moving[0]["displacement_m"]), 0.5)

    def check_shapes(self, s, n, m):
        H, W = 392, 518
        want = {
            "images": ((n, 3, H, W), torch.uint8), "K": ((n, 3, 3), torch.float32),
            "ref_T_cam": ((n, 4, 4), torch.float32), "dt": ((n,), torch.float32),
            "cam_id": ((n,), torch.int64), "slot": ((n,), torch.int64), "view": ((n,), torch.int64),
            "timestamp_ns": ((n,), torch.int64), "depth": ((n, H, W), torch.float32),
            "depth_flag": ((n, H, W), torch.uint8), "ego_mask": ((n, H, W), torch.bool),
            "box_mask": ((n, H, W), torch.uint8), "sky": ((n, H, W), torch.float32),
            "displacement_m": ((), torch.float32), "ref_T_ego": ((4, 4), torch.float32),
            "tgt_images": ((m, 3, H, W), torch.uint8), "tgt_K": ((m, 3, 3), torch.float32),
            "tgt_ref_T_cam": ((m, 4, 4), torch.float32), "tgt_dt": ((m,), torch.float32),
            "tgt_cam_id": ((m,), torch.int64), "tgt_view": ((m,), torch.int64),
            "tgt_timestamp_ns": ((m,), torch.int64), "tgt_depth": ((m, H, W), torch.float32),
            "tgt_depth_flag": ((m, H, W), torch.uint8), "tgt_ego_mask": ((m, H, W), torch.bool),
            "tgt_box_mask": ((m, H, W), torch.uint8), "tgt_sky": ((m, H, W), torch.float32),
            "tgt_offset": ((), torch.int64),
        }
        for key, (shape, dtype) in want.items():
            self.assertEqual(tuple(s[key].shape), shape, key)
            self.assertEqual(s[key].dtype, dtype, key)
        self.assertIsInstance(s["log_id"], str)
        self.assertIsInstance(s["moment"], int)

    def test_target_equals_reading_it_as_input(self):
        for mode, m in (("crop", 7), ("split", 8)):
            ds = self.ds(target_offsets=[-14, 11], front_center_mode=mode)
            n = [j for j, ((_, i), k) in enumerate(zip(ds.samples, ds.sample_offsets)) if i == 28 and k == 11][0]
            s = ds[n]
            self.check_shapes(s, 2 * m, m)
            self.assertEqual(s["tgt_cam_id"].tolist(), s["cam_id"][:m].tolist())
            self.assertEqual(s["tgt_view"].tolist(), s["view"][:m].tolist())
            self.assertTrue((s["tgt_box_mask"] & 1).any() and (s["tgt_box_mask"] & 2).any())
            self.assertTrue(s["tgt_ego_mask"].any() and (s["tgt_depth"] > 0).any())
            ref = Stage1Dataset(self.root, "val", ["fake"], input_offsets=(11,), front_center_mode=mode,
                                with_box_masks=True, with_sky=True)
            r = ref[[j for j, (_, i) in enumerate(ref.samples) if i == 28][0]]
            for key in PER_VIEW:
                self.assertTrue(torch.equal(s["tgt_" + key], r[key]), (mode, key))
        # without sky scores there is no sky key
        s = self.ds(with_sky=False)[0]
        self.assertNotIn("tgt_sky", s)
        self.assertNotIn("sky", s)


class TestHeadDatasetReal(unittest.TestCase):
    def test_real(self):
        root = _util.processed_root()
        split = os.environ.get("RV_TEST_SPLIT", "val")
        log_id = os.environ.get("RV_TEST_LOG")
        if not log_id:
            with open(os.path.join(_util.REPO, "data_lists", "av2_val_report20.txt")) as f:
                log_id = f.readline().strip()
        ds = Stage1HeadDataset(root, split, [log_id], allow_missing_logs=True)
        if len(ds) == 0:
            raise unittest.SkipTest(f"{split}/{log_id} not processed (with sky scores) under RV_PROCESSED_ROOT")
        idx = len(ds) // 2
        t0 = time.perf_counter()
        s = ds[idx]
        print(f"\n[head dataset] {log_id} sample {idx}: tgt_offset {int(s['tgt_offset'])}, "
              f"{time.perf_counter() - t0:.2f} s")
        self.assertEqual(tuple(s["images"].shape), (14, 3, 392, 518))
        self.assertEqual(tuple(s["tgt_images"].shape), (7, 3, 392, 518))
        self.assertEqual(tuple(s["tgt_sky"].shape), (7, 392, 518))
        k = int(s["tgt_offset"])
        self.assertEqual(k, ds.sample_offsets[idx])
        self.assertTrue(abs(k) <= 20 and abs(k) > 3 and abs(k + 10) > 3)
        self.assertGreaterEqual(float(s["displacement_m"]), 0.5)
        self.assertLess(float((s["tgt_dt"] - 0.05 * k).abs().max()), 0.0230)
        cov = (s["tgt_depth"] > 0).float().mean().item()
        self.assertGreater(cov, 0.05)
        pos = s["tgt_depth"][s["tgt_depth"] > 0]
        self.assertTrue(((pos >= 0.49) & (pos <= 250.2)).all())
        self.assertTrue(((s["tgt_sky"] >= 0) & (s["tgt_sky"] <= 2.56)).all())
        self.assertTrue(s["tgt_ego_mask"].any())
        R = s["tgt_ref_T_cam"][:, :3, :3].double()
        self.assertTrue(torch.allclose(R @ R.transpose(1, 2), torch.eye(3, dtype=torch.float64).expand_as(R),
                                       atol=1e-5))
        # the target cameras are where the car was, within 1 s at highway speed
        self.assertLess(float(s["tgt_ref_T_cam"][:, :3, 3].norm(dim=1).max()), 40.0)
        # the ego origin (rear axle, on the ground) lies behind and below the front-center camera
        ego_in_ref = s["ref_T_ego"][:3, 3]
        self.assertLess(float(ego_in_ref[2]), 0.0)  # z of OpenCV = forward
        self.assertGreater(float(ego_in_ref[1]), 0.0)  # y of OpenCV = down
        self.assertTrue(1.0 < float(ego_in_ref.norm()) < 4.0)


if __name__ == "__main__":
    unittest.main()
