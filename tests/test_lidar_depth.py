"""LiDAR depth on a synthetic log with known geometry and timing.

The sweeps are written as Argoverse 2 feather files: float16 xyz in the ego frame at the sweep start (i.e.
already ego-motion compensated), int32 offset_ns. Static points are sparse posts, so that a wrong transform
moves them to different pixels (a dense wall would hide a shift); a slowly moving car carries points
measured at different times inside each sweep.
"""
import os
import tempfile
import unittest

import _util  # noqa: F401
import numpy as np
import pandas as pd

from realityview.data.av2_io import PoseTable
from realityview.data.config import (FLAG_CENTER_SWEEP, FLAG_SEETHROUGH, FLAG_SYNC_DYNAMIC, default_config)
from realityview.data.geometry import make_se3, quat_to_rotmat, rotmat_to_quat, se3_inv, transform_points
from realityview.data.images import pixel_of, project
from realityview.data.lidar_depth import SweepCache, depth_for_image

SPEED = 12.0  # ego, m/s
YAW_RATE = 0.2  # rad/s
CAR_V = np.array([2.0, 0.0, 0.0])  # moving car, city frame, m/s (2 m/s * 0.1 s < the 0.3 m box margin)
T0 = 315_970_000_000_000_000  # realistic magnitude of AV2 timestamps (ns)
HW = (392, 518)
K = np.array([[427.0, 0, 258.8], [0, 427.0, 195.9], [0, 0, 1]])
# camera looking left (+y of the ego): columns are the camera x (right), y (down), z (forward) axes in ego
EGO_T_CAM = make_se3(np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]), [1.5, 0.2, 1.4])


def ego_pose(t_s):
    """city_T_ego far from the city origin (large coordinates, as in AV2)."""
    yaw = YAW_RATE * t_s
    R = quat_to_rotmat(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
    p = np.array([5000.0 + SPEED * t_s, 2000.0 + 0.5 * SPEED * YAW_RATE * t_s ** 2, 30.0])
    return make_se3(R, p)


def posts(seed=0):
    """Sparse static points beside the road, at least ~0.8 m apart, 6..40 m from the path."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < 500:
        p = np.array([5000.0 + rng.uniform(-20, 60), 2000.0 + rng.choice([-1, 1]) * rng.uniform(6, 40),
                      30.0 + rng.uniform(0.5, 6.0)])
        if not pts or np.min(np.linalg.norm(np.array(pts) - p, axis=1)) > 0.8:
            pts.append(p)
    return np.array(pts)


CAR_LWH = (4.0, 2.0, 1.5)


def car_centre(t_s):
    return np.array([5012.0, 2009.0, 30.75]) + CAR_V * t_s


def car_surface(seed=1):
    """Points on the car's side facing the road (local frame), 0.25 m apart."""
    xs = np.arange(-1.9, 1.91, 0.25)
    zs = np.arange(-0.7, 0.71, 0.25)
    a, b = np.meshgrid(xs, zs)
    return np.stack([a.ravel(), np.full(a.size, -CAR_LWH[1] / 2), b.ravel()], 1)


class SyntheticLog:
    def __init__(self, tmp, n_sweeps=16, cam_offset_ns=37_000_000, cam_sweep=7):
        self.sweep_ts = T0 + np.arange(n_sweeps, dtype=np.int64) * 100_000_000
        self.cam_ts = int(self.sweep_ts[cam_sweep] + cam_offset_ns)
        self.k0 = cam_sweep
        stamps = sorted(set(self.sweep_ts.tolist()) | {self.cam_ts})
        T = np.stack([ego_pose((s - T0) / 1e9) for s in stamps])
        q = rotmat_to_quat(T[:, :3, :3])
        self.poses = PoseTable(pd.DataFrame({"timestamp_ns": stamps, "qw": q[:, 0], "qx": q[:, 1], "qy": q[:, 2],
                                             "qz": q[:, 3], "tx_m": T[:, 0, 3], "ty_m": T[:, 1, 3],
                                             "tz_m": T[:, 2, 3]}))
        self.posts = posts()
        local = car_surface()
        rng = np.random.default_rng(2)
        self.files, rows = [], []
        self.car_meas = []  # per sweep: (city position at measurement time, measurement time)
        for ts in self.sweep_ts:
            ego_T_city = se3_inv(self.poses.city_T_ego(int(ts)))
            # static posts: one return each from each lidar
            n_s = len(self.posts)
            off_s = np.concatenate([rng.integers(0, 100_000_000, n_s), rng.integers(0, 100_000_000, n_s)])
            st = transform_points(ego_T_city, np.concatenate([self.posts, self.posts]))
            laser_s = np.concatenate([np.full(n_s, 5), np.full(n_s, 40)])
            # car: measured at ts + off, already compensated for ego motion -> the car's own motion stays
            off_c = rng.integers(0, 100_000_000, len(local))
            t_meas = (ts - T0 + off_c) / 1e9
            car_city = local + np.stack([car_centre(t) for t in t_meas])
            self.car_meas.append((car_city, ts + off_c))
            ct = transform_points(ego_T_city, car_city)
            xyz = np.concatenate([st, ct])
            off = np.concatenate([off_s, off_c])
            laser = np.concatenate([laser_s, np.full(len(ct), 7)]).astype(np.uint8)
            df = pd.DataFrame({"x": xyz[:, 0].astype(np.float16), "y": xyz[:, 1].astype(np.float16),
                               "z": xyz[:, 2].astype(np.float16), "intensity": np.zeros(len(xyz), np.uint8),
                               "laser_number": laser, "offset_ns": off.astype(np.int32)})
            f = os.path.join(tmp, f"{int(ts)}.feather")
            df.to_feather(f)
            self.files.append(f)
            # the annotation of the car at the sweep time, in the ego frame of the sweep
            c = transform_points(ego_T_city, car_centre((ts - T0) / 1e9)[None])[0]
            yaw_rel = -YAW_RATE * (ts - T0) / 1e9
            rows.append({"timestamp_ns": int(ts), "dynamic": True, "tx_m": c[0], "ty_m": c[1], "tz_m": c[2],
                         "qw": np.cos(yaw_rel / 2), "qx": 0.0, "qy": 0.0, "qz": np.sin(yaw_rel / 2),
                         "length_m": CAR_LWH[0], "width_m": CAR_LWH[1], "height_m": CAR_LWH[2]})
        self.tracks = pd.DataFrame(rows)

    def stored_ego(self, k):
        d = pd.read_feather(self.files[k])
        return d[["x", "y", "z"]].to_numpy(np.float64), d["offset_ns"].to_numpy(np.int64)


def project_ref(cam_T_ego_k, ego_pts):
    """Reference projection, written independently of the code: pixels and z of points in view."""
    pc = transform_points(cam_T_ego_k, ego_pts)
    u, v, z = project(K, pc)
    with np.errstate(invalid="ignore"):
        ok = (z >= 0.5) & (z <= 250) & (u >= -0.5) & (u < HW[1] - 0.5) & (v >= -0.5) & (v < HW[0] - 0.5)
    vi, ui = pixel_of(u[ok], v[ok])
    return vi, ui, z[ok], ok


def zmin_image(vi, ui, z):
    D = np.full(HW, np.inf)
    np.minimum.at(D, (vi, ui), z)
    return D


class TestLidarDepth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.log = SyntheticLog(cls.tmp.name)
        cls.cfg = default_config()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cache(self, cfg=None):
        cfg = cfg or self.cfg
        return SweepCache(self.log.sweep_ts, self.log.files, self.log.poses, self.log.tracks, cfg)

    def cam_T_ego(self, k):
        """Per-sweep chain: ego(sweep k) -> city (sweep pose) -> ego(camera time) -> camera."""
        lg = self.log
        return se3_inv(lg.poses.city_T_ego(lg.cam_ts) @ EGO_T_CAM) @ lg.poses.city_T_ego(int(lg.sweep_ts[k]))

    def test_single_sweep_pixel_exact(self):
        """With 0 extra sweeps, every depth pixel and z equals the reference per-sweep chain applied to the
        stored float16 xyz (converted to float64) -- no per-point offset_ns compensation."""
        lg = self.log
        cfg = default_config()
        cfg["lidar"]["sweeps_each_side"] = 0
        cfg["lidar"]["sync_sweeps_each_side"] = 0
        acc, single = depth_for_image(self.cache(cfg), lg.cam_ts, lg.k0, K, EGO_T_CAM, HW, None, cfg)
        ego, off = lg.stored_ego(lg.k0)
        n_post = 2 * len(lg.posts)
        t_meas = lg.sweep_ts[lg.k0] + off
        keep = np.ones(len(ego), bool)
        keep[n_post:] = np.abs(t_meas[n_post:] - lg.cam_ts) <= 25_000_000  # car: synchronised points only
        vi, ui, z, _ = project_ref(self.cam_T_ego(lg.k0), ego[keep])
        D = zmin_image(vi, ui, z)
        exp_v, exp_u = np.nonzero(np.isfinite(D))
        self.assertGreater(len(exp_v), 100)
        got = {(int(a), int(b)): float(c) for a, b, c in zip(acc.vu[:, 0], acc.vu[:, 1], acc.z)}
        self.assertEqual(set(got), set(zip(exp_v.tolist(), exp_u.tolist())))
        for (a, b), c in got.items():
            self.assertLessEqual(abs(c - D[a, b]) / D[a, b], 1e-3)  # float16 storage of z
        np.testing.assert_array_equal(acc.vu, single.vu)
        self.assertTrue(np.all(acc.flag & FLAG_CENTER_SWEEP))

    def test_wrong_compensation_would_be_detected(self):
        """Negative controls: no compensation, or compensating again per point with offset_ns, put the
        static points pixels away from the per-sweep result, so the exact test above has teeth."""
        lg = self.log
        k = lg.k0
        ego, off = lg.stored_ego(k)
        ego, off = ego[:len(lg.posts)], off[:len(lg.posts)]  # up-lidar posts
        _, _, _, ok = project_ref(self.cam_T_ego(k), ego)
        right = project(K, transform_points(self.cam_T_ego(k), ego[ok]))
        no_comp = project(K, transform_points(se3_inv(EGO_T_CAM), ego[ok]))
        cam_T_city = se3_inv(lg.poses.city_T_ego(lg.cam_ts) @ EGO_T_CAM)
        city_pp = np.stack([transform_points(ego_pose((lg.sweep_ts[k] + o - T0) / 1e9), p[None])[0]
                            for p, o in zip(ego[ok], off[ok])])
        per_point = project(K, transform_points(cam_T_city, city_pp))
        for wrong in (no_comp, per_point):
            d = np.hypot(wrong[0] - right[0], wrong[1] - right[1])
            self.assertGreater(np.median(d), 1.0)  # the exact test above tolerates 0 px

    def test_accumulated_layer(self):
        """+-5 sweeps of static points, synchronised car points only, flags, ranges."""
        lg = self.log
        acc, single = depth_for_image(self.cache(), lg.cam_ts, lg.k0, K, EGO_T_CAM, HW, None, self.cfg)
        n_post = 2 * len(lg.posts)
        exp_static, exp_sync, unsync = [], [], []
        for k in range(lg.k0 - 5, lg.k0 + 6):
            ego, off = lg.stored_ego(k)
            vi, ui, z, ok = project_ref(self.cam_T_ego(k), ego)
            is_car = (np.arange(len(ego)) >= n_post)[ok]
            sync = (np.abs(lg.sweep_ts[k] + off - lg.cam_ts) <= 25_000_000)[ok] & (abs(k - lg.k0) <= 1)
            for sel, dst in ((~is_car, exp_static), (is_car & sync, exp_sync), (is_car & ~sync, unsync)):
                dst.append((vi[sel], ui[sel], z[sel]))
        cat = lambda xs: tuple(np.concatenate([x[i] for x in xs]) for i in range(3))  # noqa: E731
        S, Y, U = cat(exp_static), cat(exp_sync), cat(unsync)
        D = zmin_image(np.concatenate([S[0], Y[0]]), np.concatenate([S[1], Y[1]]), np.concatenate([S[2], Y[2]]))
        exp = set(zip(*[a.tolist() for a in np.nonzero(np.isfinite(D))]))
        got = set(zip(acc.vu[:, 0].tolist(), acc.vu[:, 1].tolist()))
        self.assertEqual(got, exp)  # exactly the static + synchronised points, nearest per pixel
        # unsynchronised car points never appear (where no kept point shares their pixel)
        free = [(a, b) for a, b in zip(U[0].tolist(), U[1].tolist()) if (a, b) not in exp]
        self.assertGreater(len(free), 10)
        self.assertFalse(set(free) & got)
        # synchronised car points are flagged 2; there are some, from sweeps k0-1..k0+1 only
        sync_px = set(zip(Y[0].tolist(), Y[1].tolist()))
        f2 = {(int(a), int(b)) for a, b, f in zip(acc.vu[:, 0], acc.vu[:, 1], acc.flag) if f & FLAG_SYNC_DYNAMIC}
        self.assertGreater(len(f2), 5)
        self.assertTrue(f2 <= sync_px)
        # the single layer is k0's static points plus the same synchronised points (sparse posts mostly
        # re-project onto the same pixels, so accumulation adds few pixels here)
        self.assertGreater(len(acc), len(single))
        self.assertTrue(np.all((single.flag & (FLAG_CENTER_SWEEP | FLAG_SYNC_DYNAMIC)) > 0))
        z = acc.z.astype(np.float64)
        self.assertTrue(np.all((z >= 0.5) & (z <= 250)))
        self.assertTrue(np.all(acc.vu[:, 0] < HW[0]) and np.all(acc.vu[:, 1] < HW[1]))
        # flag 4 is exactly "depth > 1.2 x the 5x5 minimum"
        Dacc = np.full(HW, np.inf)
        Dacc[acc.vu[:, 0], acc.vu[:, 1]] = acc.z.astype(np.float32)
        pad = np.pad(Dacc, 2, constant_values=np.inf)
        mn = np.min([pad[dv:dv + HW[0], du:du + HW[1]] for dv in range(5) for du in range(5)], axis=0)
        st = Dacc[acc.vu[:, 0], acc.vu[:, 1]] > 1.2 * mn[acc.vu[:, 0], acc.vu[:, 1]]
        np.testing.assert_array_equal(st, (acc.flag & FLAG_SEETHROUGH) > 0)

    def test_ego_mask_drops_points(self):
        lg = self.log
        mask = np.zeros(HW, bool)
        mask[:, :259] = True
        acc, single = depth_for_image(self.cache(), lg.cam_ts, lg.k0, K, EGO_T_CAM, HW, mask, self.cfg)
        full, _ = depth_for_image(self.cache(), lg.cam_ts, lg.k0, K, EGO_T_CAM, HW, None, self.cfg)
        self.assertGreater(len(acc), 50)
        self.assertFalse(mask[acc.vu[:, 0], acc.vu[:, 1]].any())
        self.assertFalse(mask[single.vu[:, 0], single.vu[:, 1]].any())
        self.assertGreater(mask[full.vu[:, 0], full.vu[:, 1]].sum(), 50)


class TestRealLidar(unittest.TestCase):
    def test_two_lidars_real_log(self):
        """On a real log, the two LiDARs (and consecutive sweeps) agree on static surfaces once moved to the
        camera time with the per-sweep chain (design check 5: median nearest-neighbour distance <= 0.15 m)."""
        from realityview.data.checks import check_lidar_two_sensor
        from realityview.data.pipeline import LogContext

        path, _, _ = _util.real_log()
        ctx = LogContext(path, default_config(), None)
        r = check_lidar_two_sensor(ctx)
        self.assertTrue(r["passed"], {k: v for k, v in r.items() if k != "per_sweep"})


if __name__ == "__main__":
    unittest.main()
