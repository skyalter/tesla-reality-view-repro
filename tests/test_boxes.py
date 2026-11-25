import unittest

import _util  # noqa: F401
import numpy as np
import pandas as pd

from realityview.data.boxes import (MASK_ANY, MASK_MOVING, MASK_PERSON, draw_box_mask, estimate_motion,
                                    interpolate_boxes, points_in_boxes)
from realityview.data.config import default_config
from realityview.data.geometry import make_se3, quat_to_rotmat, se3_inv

T0 = 1_000_000_000_000


def track(uuid, category, n, pos_fn, yaw_fn, dims=(4.5, 1.9, 1.6), start=0):
    rows = []
    for k in range(start, start + n):
        t = k * 0.1
        yaw = yaw_fn(t)
        p = pos_fn(t)
        rows.append({"timestamp_ns": T0 + k * 100_000_000, "track_uuid": uuid, "category": category,
                     "cx": p[0], "cy": p[1], "cz": p[2], "cqw": np.cos(yaw / 2), "cqx": 0.0, "cqy": 0.0,
                     "cqz": np.sin(yaw / 2), "length_m": dims[0], "width_m": dims[1], "height_m": dims[2]})
    return rows


def make_tracks():
    rows = []
    rows += track("car", "REGULAR_VEHICLE", 30, lambda t: (100 + 10 * t, 50.0, 1.0), lambda t: 0.5 * t)
    rows += track("parked", "REGULAR_VEHICLE", 30, lambda t: (120.0, 55.0, 1.0), lambda t: 0.0)
    rows += track("ped", "PEDESTRIAN", 30, lambda t: (110.0, 48.0, 1.0), lambda t: 0.0, dims=(0.7, 0.7, 1.7))
    rows += track("short", "REGULAR_VEHICLE", 2, lambda t: (130.0, 60.0, 1.0), lambda t: 0.0, start=10)
    return estimate_motion(pd.DataFrame(rows), default_config())


class TestMotion(unittest.TestCase):
    def setUp(self):
        self.t = make_tracks()

    def test_speed_and_flags(self):
        car = self.t[self.t.track_uuid == "car"]
        np.testing.assert_allclose(car["speed"], 10.0, rtol=1e-9)
        self.assertTrue(car["moving"].all())
        parked = self.t[self.t.track_uuid == "parked"]
        self.assertTrue((parked["speed"] < 1e-9).all())
        self.assertFalse(parked["moving"].any())
        self.assertFalse(parked["dynamic"].any())
        ped = self.t[self.t.track_uuid == "ped"]
        self.assertFalse(ped["moving"].any())
        self.assertTrue(ped["person_like"].all() and ped["dynamic"].all())
        short = self.t[self.t.track_uuid == "short"]  # 0.1 s of track: speed unknown -> moving
        self.assertTrue(short["speed"].isna().all())
        self.assertTrue(short["moving"].all())


class TestMotionNoise(unittest.TestCase):
    def test_jittery_parked_car_is_static(self):
        """Box centres jittering by +-0.06 m (0.1 s apart this is up to 1.2 m/s) are not moving: speed comes
        from centres ~1 s apart."""
        rng = np.random.default_rng(0)
        rows = track("parked", "REGULAR_VEHICLE", 50, lambda t: (120.0 + rng.uniform(-0.06, 0.06),
                                                                  55.0 + rng.uniform(-0.06, 0.06), 1.0),
                     lambda t: 0.0)
        t = estimate_motion(pd.DataFrame(rows), default_config())
        self.assertLess(t["speed"].max(), 0.2)
        self.assertFalse(t["moving"].any())


class TestInterpolation(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config()
        self.t = make_tracks()

    def query(self, ts):
        b = interpolate_boxes(self.t, np.array(ts, np.int64), self.cfg)
        return b

    def test_linear_centre_and_slerp(self):
        ts = T0 + 1_234_000_000  # between annotations 12 and 13
        b = self.query([ts])
        car = b[b.track_uuid == "car"].iloc[0]
        self.assertAlmostEqual(car.cx, 100 + 10 * 1.234, places=9)
        yaw = 2 * np.arctan2(car.cqz, car.cqw)
        self.assertAlmostEqual(yaw, 0.5 * 1.234, places=9)
        self.assertFalse(car.extrapolated)

    def test_extrapolation_limit(self):
        last = T0 + 29 * 100_000_000
        b = self.query([last + 40_000_000, last + 60_000_000, T0 - 50_000_000, T0 - 51_000_000])
        car = b[b.track_uuid == "car"]
        self.assertEqual(sorted(car["q"].tolist()), [0, 2])
        row = car[car.q == 0].iloc[0]
        self.assertTrue(row.extrapolated)
        self.assertAlmostEqual(row.cx, 100 + 10 * 2.94, places=6)  # moved along the end velocity

    def test_slerp_halfway_and_gap(self):
        """Orientation is interpolated on the sphere (yaw 0.5 rad/s: halfway between two annotations)."""
        b = self.query([T0 + 1_250_000_000])
        car = b[b.track_uuid == "car"].iloc[0]
        self.assertAlmostEqual(2 * np.arctan2(car.cqz, car.cqw), 0.625, places=9)
        self.assertAlmostEqual(car.cqw ** 2 + car.cqz ** 2, 1.0, places=12)
        # a gap longer than max_gap_s inside a track is not bridged
        t = self.t[~((self.t.track_uuid == "parked") & self.t.timestamp_ns.between(T0 + 1_000_000_000,
                                                                                    T0 + 1_500_000_000))]
        b = interpolate_boxes(t, np.array([T0 + 1_250_000_000], np.int64), self.cfg)
        self.assertNotIn("parked", set(b.track_uuid))

    def test_exact_hit(self):
        b = self.query([T0 + 29 * 100_000_000])
        car = b[b.track_uuid == "car"].iloc[0]
        self.assertFalse(car.extrapolated)
        self.assertAlmostEqual(car.cx, 129.0, places=9)


class TestBoxGeometry(unittest.TestCase):
    def test_points_in_boxes(self):
        """Margins apply on every side: a 4 x 2 x 2 box with 0.3 / 0.2 m is tested as 4.6 x 2.6 x 2.4."""
        yaw = np.pi / 4
        boxes = pd.DataFrame([{"tx_m": 10.0, "ty_m": 0.0, "tz_m": 1.0, "qw": np.cos(yaw / 2), "qx": 0.0, "qy": 0.0,
                               "qz": np.sin(yaw / 2), "length_m": 4.0, "width_m": 2.0, "height_m": 2.0}])
        R = quat_to_rotmat(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        local = np.array([[0, 0, 0], [2.29, 0, 0], [2.31, 0, 0], [0, 1.29, 0], [0, -1.31, 0], [0, 0, 1.19],
                          [0, 0, -1.21], [-2.29, -1.29, 1.19]])
        pts = local @ R.T + [10.0, 0.0, 1.0]
        inside = points_in_boxes(pts, boxes, margin_lw=0.3, margin_h=0.2)
        np.testing.assert_array_equal(inside, [True, True, False, True, False, True, False, True])

    def test_mask_drawn_and_partial(self):
        cfg = default_config()
        K = np.array([[427.0, 0, 258.8], [0, 427.0, 195.9], [0, 0, 1]])
        # camera at the city origin looking along +x (cam z = city x, cam x = -city y, cam y = -city z)
        R = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
        cam_T_city = se3_inv(make_se3(R, [0, 0, 0]))

        def box(x, y, moving=True, person=False):
            return {"cx": x, "cy": y, "cz": 0.0, "cqw": 1.0, "cqx": 0.0, "cqy": 0.0, "cqz": 0.0, "length_m": 4.0,
                    "width_m": 2.0, "height_m": 1.5, "vx": 5.0, "vy": 0.0, "vz": 0.0, "moving": moving,
                    "person_like": person}

        m = draw_box_mask(pd.DataFrame([box(15.0, 0.0)]), cam_T_city, K, (518, 392), cfg)
        self.assertEqual(m[196, 259], MASK_ANY | MASK_MOVING)
        self.assertEqual(m[5, 5], 0)
        # a box beside the camera, partly behind it and partly out of frame, is still drawn
        m2 = draw_box_mask(pd.DataFrame([box(1.0, -2.0, moving=False, person=True)]), cam_T_city, K, (518, 392), cfg)
        self.assertTrue((m2 & MASK_PERSON).any())
        self.assertFalse((m2 & MASK_MOVING).any())
        # a box fully behind the camera draws nothing
        m3 = draw_box_mask(pd.DataFrame([box(-15.0, 0.0)]), cam_T_city, K, (518, 392), cfg)
        self.assertFalse(m3.any())
        # bits restricts what is drawn
        m4 = draw_box_mask(pd.DataFrame([box(15.0, 0.0)]), cam_T_city, K, (518, 392), cfg, bits=MASK_PERSON)
        self.assertFalse(m4.any())

    def test_mask_covers_every_visible_point(self):
        """Random boxes, many partly behind the camera or outside the image: every point of the grown,
        swept box that is in front of the camera and projects into the image lies in the mask, and the mask
        does not reach far beyond those points."""
        from scipy.spatial import Delaunay

        from realityview.data.boxes import box_corners_city

        cfg = default_config()
        cfg["masks"]["dilate_px"] = 0
        K = np.array([[427.0, 0, 258.8], [0, 427.0, 195.9], [0, 0, 1]])
        R = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
        cam_T_city = se3_inv(make_se3(R, [0, 0, 0]))
        rng = np.random.default_rng(0)
        n_partial = 0
        for _ in range(60):
            yaw = rng.uniform(-np.pi, np.pi)
            b = pd.DataFrame([{"cx": rng.uniform(-3, 20), "cy": rng.uniform(-8, 8), "cz": rng.uniform(-1, 1),
                               "cqw": np.cos(yaw / 2), "cqx": 0.0, "cqy": 0.0, "cqz": np.sin(yaw / 2),
                               "length_m": rng.uniform(0.5, 10), "width_m": rng.uniform(0.5, 3),
                               "height_m": rng.uniform(1, 3), "vx": rng.uniform(-8, 8), "vy": rng.uniform(-2, 2),
                               "vz": 0.0, "moving": True, "person_like": False}])
            m = draw_box_mask(b, cam_T_city, K, (518, 392), cfg)
            C = box_corners_city(b, cfg)[0]
            lo, hi = C.min(0), C.max(0)
            P = rng.uniform(lo, hi, (200000, 3))
            P = P[Delaunay(C).find_simplex(P) >= 0]
            pc = P @ cam_T_city[:3, :3].T + cam_T_city[:3, 3]
            n_partial += int((pc[:, 2] < 0).any() and (pc[:, 2] > 0.5).any())
            pc = pc[pc[:, 2] > 0.06]
            u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
            v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
            ok = (u >= -0.5) & (u < 517.5) & (v >= -0.5) & (v < 391.5)
            if not ok.any():
                continue
            ui, vi = np.floor(u[ok] + 0.5).astype(int), np.floor(v[ok] + 0.5).astype(int)
            hit = np.zeros((392, 518), bool)
            hit[vi, ui] = True
            missed = hit & (m == 0)
            # only single border pixels may differ (pixel-centre rounding on the outline)
            self.assertLessEqual(int(missed.sum()), 2)
        self.assertGreater(n_partial, 5)

    def test_mask_partial_box_exact_outline(self):
        """A long box beside the camera, from behind it to 10 m ahead: the mask reaches the image border."""
        cfg = default_config()
        cfg["masks"]["dilate_px"] = 0
        cfg["masks"]["velocity_stretch_s"] = 0.0
        K = np.array([[427.0, 0, 258.8], [0, 427.0, 195.9], [0, 0, 1]])
        R = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
        cam_T_city = se3_inv(make_se3(R, [0, 0, 0]))
        b = pd.DataFrame([{"cx": 2.5, "cy": -3.0, "cz": 0.0, "cqw": 1.0, "cqx": 0.0, "cqy": 0.0, "cqz": 0.0,
                           "length_m": 15.0, "width_m": 1.4, "height_m": 1.4, "vx": 0.0, "vy": 0.0, "vz": 0.0,
                           "moving": True, "person_like": False}])
        m = draw_box_mask(b, cam_T_city, K, (518, 392), cfg)
        self.assertTrue(m[196, 517])  # right image border, at the horizon row
        # far end at x = 10 + 0.3: its left edge (y = -3 + 1.0) projects to u = 258.8 + 427 * 2.0 / 10.3
        u_far = 258.8 + 427 * (3.0 - 0.7 - 0.3) / (10.0 + 0.3)
        self.assertTrue(m[196, int(np.ceil(u_far)) + 1])
        self.assertFalse(m[196, int(np.floor(u_far)) - 2])


class TestRealBoxes(unittest.TestCase):
    def test_static_tracks_not_moving(self):
        from realityview.data import av2_io
        from realityview.data.boxes import prepare_tracks
        from realityview.data.checks import check_motion

        path, _, _ = _util.real_log()
        cfg = default_config()
        tracks = prepare_tracks(av2_io.read_annotations(path), av2_io.read_poses(path), cfg)
        self.assertIn("track_uuid", tracks.columns)
        r = check_motion(tracks, cfg["boxes"]["moving_speed_mps"])
        self.assertTrue(r["passed"], r)


if __name__ == "__main__":
    unittest.main()
