import unittest

import _util  # noqa: F401
import numpy as np
import pandas as pd

from realityview.data.av2_io import PoseTable
from realityview.data.config import CAMS, FRONT_CENTER, default_config
from realityview.data.index import build_frames_table, build_moments_table, nearest_index, sweep_before

OFFSETS_MS = {"ring_front_center": 0.0, "ring_front_left": -12.5, "ring_front_right": 12.5, "ring_side_left": 22.5,
              "ring_side_right": -22.4, "ring_rear_left": 7.5, "ring_rear_right": -7.5}
T0 = 500_000_000_000


def synthetic(n=60, drop_first_side_left=True, stop_after=None):
    cam_ts = {}
    for cam in CAMS:
        ts = T0 + 40_000_000 + (np.arange(n) * 50_000_000 + OFFSETS_MS[cam] * 1e6).astype(np.int64)
        if cam == "ring_side_left" and drop_first_side_left:
            ts = ts[1:]  # first frame missing -> moment 0 incomplete
        cam_ts[cam] = ts
    sweep_ts = T0 + np.arange(n // 2 + 2, dtype=np.int64) * 100_000_000
    stamps = np.unique(np.concatenate(list(cam_ts.values()) + [sweep_ts]))
    t = (stamps - T0) / 1e9
    x = 5.0 * t if stop_after is None else 5.0 * np.minimum(t, stop_after)
    poses = PoseTable(pd.DataFrame({"timestamp_ns": stamps, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
                                    "tx_m": x, "ty_m": 0.0, "tz_m": 0.0}))
    return cam_ts, sweep_ts, poses


class TestIndex(unittest.TestCase):
    def test_nearest_and_sweep_before(self):
        a = np.array([0, 10, 20, 30])
        np.testing.assert_array_equal(nearest_index(a, np.array([-5, 4, 5, 6, 26, 99])), [0, 0, 0, 1, 3, 3])
        np.testing.assert_array_equal(sweep_before(a, np.array([-1, 0, 9, 10, 35])), [-1, 0, 0, 1, 3])

    def test_moments(self):
        cfg = default_config()
        cam_ts, sweep_ts, poses = synthetic()
        m = build_moments_table(cam_ts, poses, sweep_ts, cfg)
        n = len(cam_ts[FRONT_CENTER])
        self.assertEqual(len(m), n)
        self.assertFalse(m["complete"].iloc[0])  # side-left frame 0 missing -> nearest is 72.5 ms away
        self.assertTrue(m["complete"].iloc[1:].all())
        self.assertAlmostEqual(m["max_abs_dt_ms"].iloc[5], 22.5, places=6)
        for cam in CAMS:
            dt = m[f"dt_ms_{cam}"].iloc[1:]
            np.testing.assert_allclose(dt, OFFSETS_MS[cam], atol=1e-6)
        v1 = m["valid_stage1"].to_numpy()
        # i >= 10, t complete and t-10 complete; moment 10 has t-10 = 0 incomplete
        self.assertFalse(v1[:11].any())
        self.assertTrue(v1[11:].all())
        v2 = m["valid_stage2"].to_numpy()
        self.assertTrue(v2[20:n - 20].all())
        self.assertFalse(v2[:20].any() or v2[n - 20:].any())
        np.testing.assert_allclose(m["disp_t05_m"].iloc[10:], 2.5, atol=1e-9)  # 5 m/s * 0.5 s
        self.assertAlmostEqual(m["disp_win_max_m"].iloc[30], 5.0, places=9)  # 1 s window either side
        # every image row of the frames table has its exact pose and a sweep that started before it
        f = build_frames_table(cam_ts, poses, sweep_ts)
        self.assertEqual(len(f), sum(len(v) for v in cam_ts.values()))
        ok = f["k0_before_first_sweep"] | (sweep_ts[f["k0"].to_numpy()] <= f["timestamp_ns"].to_numpy())
        self.assertTrue(ok.all())

    def test_missing_pose_raises(self):
        cam_ts, sweep_ts, poses = synthetic()
        cam_ts[FRONT_CENTER] = cam_ts[FRONT_CENTER] + 1
        with self.assertRaises(KeyError):
            build_moments_table(cam_ts, poses, sweep_ts, default_config())


class TestTimestampCheck(unittest.TestCase):
    def ctx(self, cam_ts, sweep_ts, poses):
        class Ctx:
            pass

        c = Ctx()
        c.cam_ts, c.sweep_ts, c.poses = cam_ts, sweep_ts, poses
        return c

    def test_edges_allowed_interior_not(self):
        from realityview.data.checks import check_timestamps

        cfg = default_config()
        n = 320
        cam_ts, sweep_ts, poses = synthetic(n=n)  # first side-left frame missing: moment 0 incomplete
        m = build_moments_table(cam_ts, poses, sweep_ts, cfg)
        r = check_timestamps(self.ctx(cam_ts, sweep_ts, poses), m)
        self.assertTrue(r["passed"], r)
        self.assertEqual(r["n_incomplete"], 1)
        # one side-left frame missing in the middle of the log: its moment is incomplete, which is tolerated
        cam_ts["ring_side_left"] = np.delete(cam_ts["ring_side_left"], 150)
        m = build_moments_table(cam_ts, poses, sweep_ts, cfg)
        r = check_timestamps(self.ctx(cam_ts, sweep_ts, poses), m)
        self.assertTrue(r["passed"], r)
        self.assertGreaterEqual(r["n_incomplete_interior"], 1)
        # frames missing all over the log -> fails
        cam_ts["ring_side_left"] = np.delete(cam_ts["ring_side_left"], [40, 80, 120, 200, 240])
        m = build_moments_table(cam_ts, poses, sweep_ts, cfg)
        r = check_timestamps(self.ctx(cam_ts, sweep_ts, poses), m)
        self.assertFalse(r["passed"])
        self.assertGreater(r["n_incomplete_interior"], 3)
        # three frames in a row missing -> fails on the gap
        cam_ts, sweep_ts, poses = synthetic(n=n)
        cam_ts["ring_side_left"] = np.delete(cam_ts["ring_side_left"], [150, 151, 152])
        m = build_moments_table(cam_ts, poses, sweep_ts, cfg)
        self.assertFalse(check_timestamps(self.ctx(cam_ts, sweep_ts, poses), m)["passed"])

    def test_rejudge_applies_current_limits(self):
        from realityview.data.checks import rejudge_log_checks

        doc = {"passed": False, "checks": {
            "motion": {"passed": False, "still_judged_moving_frac": 0.03, "threshold": "old"},
            "lidar_two_sensor": {"passed": True, "up_vs_down_max_m": 0.5, "sweep_vs_next_max_m": 0.05},
            "boxes": {"passed": True}}}
        doc, changed = rejudge_log_checks(doc)
        self.assertEqual(sorted(changed), ["lidar_two_sensor", "motion"])
        self.assertTrue(doc["checks"]["motion"]["passed"])
        self.assertFalse(doc["checks"]["lidar_two_sensor"]["passed"])
        self.assertFalse(doc["passed"])
        doc["checks"]["lidar_two_sensor"]["up_vs_down_max_m"] = 0.1
        doc, changed = rejudge_log_checks(doc)
        self.assertTrue(doc["passed"])

class TestRealIndex(unittest.TestCase):
    def test_real_log(self):
        from realityview.data import av2_io

        path, _, _ = _util.real_log()
        cfg = default_config()
        cam_ts = {c: av2_io.list_camera_frames(path, c)[0] for c in CAMS}
        sweep_ts, _ = av2_io.list_sweeps(path)
        m = build_moments_table(cam_ts, av2_io.read_poses(path), sweep_ts, cfg)
        self.assertLessEqual(m.loc[m["complete"], "max_abs_dt_ms"].max(), 25.0)
        self.assertGreaterEqual(int(m["valid_stage2"].sum()), 270)
        self.assertGreaterEqual(int(m["valid_stage1"].sum()), 300)
        v = m[m["valid_stage1"]]
        self.assertTrue((v["i"] >= 10).all())


if __name__ == "__main__":
    unittest.main()
