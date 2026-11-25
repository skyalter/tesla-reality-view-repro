import os
import tempfile
import unittest

import _util  # noqa: F401
import numpy as np
import pandas as pd

from realityview.data.av2_io import PoseTable
from realityview.data.config import default_config
from realityview.data.geometry import make_se3, quat_to_rotmat, rotmat_to_quat, se3_inv, transform_points
from realityview.data.lidar_depth import SweepCache, two_lidar_consistency

SPEED = 15.0  # m/s
YAW_RATE = 0.3  # rad/s


def ego_pose(t_s):
    """city_T_ego of a car driving a circle-ish path, far from the city origin (large coordinates)."""
    yaw = YAW_RATE * t_s
    R = quat_to_rotmat(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
    p = np.array([5000.0 + SPEED * t_s * np.cos(yaw / 2), 2000.0 + SPEED * t_s * np.sin(yaw / 2), 30.0])
    return make_se3(R, p)


def world_points(seed=0):
    """Two walls and some posts, all static, in the city frame near the car path."""
    rng = np.random.default_rng(seed)
    n = 6000
    x = 5000.0 + rng.uniform(-10, 40, n)
    side = rng.choice([-8.0, 12.0], n)
    y = 2000.0 + side + rng.normal(0, 0.01, n)
    z = 30.0 + rng.uniform(0.7, 4.0, n)
    return np.stack([x, y, z], 1)


class SyntheticLog:
    """Sweeps written as AV2 feather files: points already compensated to the sweep start time."""

    def __init__(self, tmp, n_sweeps=8):
        self.t0 = 10_000_000_000
        self.sweep_ts = self.t0 + np.arange(n_sweeps, dtype=np.int64) * 100_000_000
        self.cam_ts = self.sweep_ts[3] + 37_000_000  # a camera exposed 37 ms into sweep 3
        stamps = sorted(set(self.sweep_ts.tolist()) | {int(self.cam_ts)})
        T = np.stack([ego_pose((s - self.t0) / 1e9) for s in stamps])
        q = rotmat_to_quat(T[:, :3, :3])
        self.poses = PoseTable(pd.DataFrame({"timestamp_ns": stamps, "qw": q[:, 0], "qx": q[:, 1], "qy": q[:, 2],
                                             "qz": q[:, 3], "tx_m": T[:, 0, 3], "ty_m": T[:, 1, 3], "tz_m": T[:, 2, 3]}))
        self.world = world_points()
        self.files = []
        rng = np.random.default_rng(1)
        for ts in self.sweep_ts:
            # each lidar samples the same surfaces at its own points (the down lidar sees the walls shifted
            # along them by up to 5 cm, as two real scanners would)
            jitter = np.zeros_like(self.world)
            jitter[:, 0] = rng.uniform(-0.05, 0.05, len(self.world))
            ego = transform_points(se3_inv(self.poses.city_T_ego(int(ts))),
                                   np.concatenate([self.world, self.world + jitter]))
            n = len(ego)
            laser = np.where(np.arange(n) < len(self.world), 5, 40).astype(np.uint8)  # up, then down lidar
            # measurement times: the down lidar is ~50 ms out of phase with the up lidar
            off = rng.uniform(0, 50e6, n).astype(np.int64) + np.where(laser >= 32, 50_000_000, 0)
            df = pd.DataFrame({"x": ego[:, 0].astype(np.float16), "y": ego[:, 1].astype(np.float16),
                               "z": ego[:, 2].astype(np.float16), "intensity": np.zeros(n, np.uint8),
                               "laser_number": laser, "offset_ns": off.astype(np.int32)})
            f = os.path.join(tmp, f"{int(ts)}.feather")
            df.to_feather(f)
            self.files.append(f)
        self.tracks = pd.DataFrame({"timestamp_ns": pd.Series([], dtype=np.int64), "dynamic": pd.Series([], dtype=bool),
                                    "tx_m": [], "ty_m": [], "tz_m": [], "qw": [], "qx": [], "qy": [], "qz": [],
                                    "length_m": [], "width_m": [], "height_m": []})


class TestPerSweepCompensation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.log = SyntheticLog(cls.tmp.name)
        cls.cfg = default_config()
        cls.cache = SweepCache(cls.log.sweep_ts, cls.log.files, cls.log.poses, cls.log.tracks, cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_city_points_recovered(self):
        """Sweep points moved to the city frame with the sweep-time pose are the static world points
        (up to the float16 storage step)."""
        for k in (0, 3, 7):
            city = self.cache.get(k)["city"]
            self.assertLess(np.abs(city[:len(self.log.world)] - self.log.world).max(), 0.04)

    def test_two_lidars_agree_at_camera_time(self):
        """The consistency check (design check 5) passes on correctly compensated sweeps; the real-data
        version is in test_lidar_depth.py, the pixel-exact transform test as well."""
        r = two_lidar_consistency(self.cache, 3, int(self.log.cam_ts))
        self.assertLessEqual(r["up_vs_down_median_m"], 0.15)
        self.assertLessEqual(r["sweep_vs_next_median_m"], 0.15)

    def test_per_point_compensation_would_be_wrong(self):
        """Compensating again with offset_ns (the mistake in the first design) separates the two lidars."""
        from scipy.spatial import cKDTree

        sw = pd.read_feather(self.log.files[3])
        ego = sw[["x", "y", "z"]].to_numpy(np.float64)
        t = self.log.sweep_ts[3] + sw["offset_ns"].to_numpy(np.int64)
        city = np.stack([transform_points(ego_pose((ti - self.log.t0) / 1e9), p[None])[0] for ti, p in zip(t, ego)])
        up = sw["laser_number"].to_numpy() < 32
        d, _ = cKDTree(city[up]).query(city[~up])
        self.assertGreater(np.median(d), 0.15)


if __name__ == "__main__":
    unittest.main()
