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

    def test_exact_hit(self):
        b = self.query([T0 + 29 * 100_000_000])
        car = b[b.track_uuid == "car"].iloc[0]
        self.assertFalse(car.extrapolated)
        self.assertAlmostEqual(car.cx, 129.0, places=9)


class TestBoxGeometry(unittest.TestCase):
    def test_points_in_boxes(self):
        boxes = pd.DataFrame([{"tx_m": 10.0, "ty_m": 0.0, "tz_m": 1.0, "qw": np.cos(np.pi / 8), "qx": 0.0, "qy": 0.0,
                               "qz": np.sin(np.pi / 8), "length_m": 4.0, "width_m": 2.0, "height_m": 2.0}])
        pts = np.array([[10.0, 0.0, 1.0], [10 + 1.9 * np.cos(np.pi / 4), 1.9 * np.sin(np.pi / 4), 1.0],
                        [10.0, 1.9, 1.0], [10.0, 0.0, 2.05], [10.0, 0.0, 2.2]])
        inside = points_in_boxes(pts, boxes, margin_lw=0.3, margin_h=0.2)
        np.testing.assert_array_equal(inside, [True, True, False, True, False])

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
        R_check = quat_to_rotmat(np.array([1.0, 0, 0, 0]))
        np.testing.assert_allclose(R_check, np.eye(3))


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
