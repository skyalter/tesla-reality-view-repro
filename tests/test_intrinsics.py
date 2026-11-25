import unittest

import _util  # noqa: F401  (sets sys.path)
import cv2
import numpy as np

from realityview.data.config import default_config
from realityview.data.images import (crop_intrinsics, front_center_view, intrinsics_for_half_pixel_renderer,
                                     pixel_of, project, scale_intrinsics)

K_LAND = np.array([[1686.46, 0.0, 1025.50], [0.0, 1686.46, 776.53], [0.0, 0.0, 1.0]])
K_FC = np.array([[1781.49, 0.0, 775.51], [0.0, 1781.49, 1022.94], [0.0, 0.0, 1.0]])


def random_points(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(-30, 30, n), rng.uniform(-8, 8, n), rng.uniform(1, 120, n)], 1)


class TestScaling(unittest.TestCase):
    def test_same_point_same_place(self):
        """Projecting with the scaled K equals projecting at full resolution and mapping to the small grid."""
        pts = random_points()
        for K, full, small in ((K_LAND, (2048, 1550), (518, 392)), (K_FC, (1550, 2048), (518, 684))):
            u0, v0, _ = project(K, pts)
            Ks = scale_intrinsics(K, full, small)
            u1, v1, _ = project(Ks, pts)
            sx, sy = small[0] / full[0], small[1] / full[1]
            np.testing.assert_allclose(u1, (u0 + 0.5) * sx - 0.5, atol=1e-9)
            np.testing.assert_allclose(v1, (v0 + 0.5) * sy - 0.5, atol=1e-9)

    def test_image_corners_map_to_corners(self):
        """The full-image extent [-0.5, W-0.5] maps onto the small-image extent [-0.5, W'-0.5]."""
        Ks = scale_intrinsics(K_LAND, (2048, 1550), (518, 392))
        Kinv, Ksinv = np.linalg.inv(K_LAND), np.linalg.inv(Ks)
        for (u, v), (us, vs) in (((-0.5, -0.5), (-0.5, -0.5)), ((2047.5, 1549.5), (517.5, 391.5))):
            ray_full = Kinv @ [u, v, 1.0]
            ray_small = Ksinv @ [us, vs, 1.0]
            np.testing.assert_allclose(ray_full, ray_small, atol=1e-12)

    def test_resize_moves_content_like_k(self):
        """A bright block drawn at full resolution lands, after INTER_AREA, where the scaled K predicts."""
        cfg = default_config()
        full, small = (2048, 1550), (518, 392)
        Ks = scale_intrinsics(K_LAND, full, small)
        for p in ([3.0, 1.0, 20.0], [-6.0, -2.0, 15.0], [0.37, 0.11, 9.0]):
            u0, v0, _ = project(K_LAND, np.array([p]))
            img = np.zeros((full[1], full[0]), np.float32)
            r, c = int(np.floor(v0[0])), int(np.floor(u0[0]))
            img[r - 7:r + 9, c - 7:c + 9] = 1.0  # 16x16 block, pixel-centre centroid at (r + 0.5, c + 0.5)
            cu, cv_ = c + 0.5, r + 0.5
            sm = cv2.resize(img, small, interpolation=cv2.INTER_AREA)
            yy, xx = np.mgrid[0:small[1], 0:small[0]]
            mu, mv = (sm * xx).sum() / sm.sum(), (sm * yy).sum() / sm.sum()
            # centroid in the small image vs. the full-res centroid mapped with the same rule as K
            self.assertAlmostEqual(mu, (cu + 0.5) * small[0] / full[0] - 0.5, delta=0.02)
            self.assertAlmostEqual(mv, (cv_ + 0.5) * small[1] / full[1] - 0.5, delta=0.02)
        self.assertEqual(cfg["images"]["landscape_wh"], [518, 392])

    def test_pixel_index_convention(self):
        r, c = pixel_of(np.array([-0.5, -0.49, 0.49, 0.5, 517.49]), np.array([-0.5, 0.0, 0.49, 0.5, 391.49]))
        np.testing.assert_array_equal(c, [0, 0, 0, 1, 517])
        np.testing.assert_array_equal(r, [0, 0, 0, 1, 391])

    def test_half_pixel_renderer(self):
        K = intrinsics_for_half_pixel_renderer(K_LAND)
        self.assertAlmostEqual(K[0, 2], K_LAND[0, 2] + 0.5)
        self.assertAlmostEqual(K[1, 2], K_LAND[1, 2] + 0.5)


class TestFrontCenter(unittest.TestCase):
    def test_crop_rows_and_k(self):
        cfg = default_config()
        Ks = scale_intrinsics(K_FC, (1550, 2048), (518, 684))
        img = np.arange(684 * 518, dtype=np.int64).reshape(684, 518)
        out, Kc = front_center_view(img, Ks, "crop", cfg)
        self.assertEqual(out.shape, (392, 518))
        self.assertEqual(out[0, 0], img[146, 0])
        pts = random_points()
        u, v, _ = project(Ks, pts)
        uc, vc, _ = project(Kc, pts)
        np.testing.assert_allclose(vc, v - 146, atol=1e-9)
        np.testing.assert_allclose(uc, u, atol=1e-9)
        # rows 146..538 of 518x684 correspond to full-resolution rows ~437..1610
        self.assertAlmostEqual((146 - 0.5 + 0.5) / (684 / 2048) - 0.5, 436.7, delta=0.2)
        self.assertAlmostEqual((538 - 0.5 + 0.5) / (684 / 2048) - 0.5, 1610.4, delta=0.2)

    def test_crop_intrinsics(self):
        K = crop_intrinsics(K_FC, row0=10, col0=3)
        self.assertAlmostEqual(K[1, 2], K_FC[1, 2] - 10)
        self.assertAlmostEqual(K[0, 2], K_FC[0, 2] - 3)

    def test_reserved_modes(self):
        cfg = default_config()
        img = np.zeros((684, 518, 3), np.uint8)
        for mode in ("split", "letterbox"):
            with self.assertRaises(ValueError):
                front_center_view(img, K_FC, mode, cfg)  # single-view helper: crop only
        with self.assertRaises(ValueError):
            front_center_view(img, K_FC, "nope", cfg)


class TestFrontCenterViews(unittest.TestCase):
    """The three ways of turning the stored 518x684 front-center image into 518x392 input."""

    def setUp(self):
        self.cfg = default_config()
        self.Ks = scale_intrinsics(K_FC, (1550, 2048), (518, 684))

    def views(self, mode):
        from realityview.data.images import front_center_views

        return front_center_views(mode, self.cfg)

    def test_view_k_matches_image_geometry(self):
        """Projecting with the view K equals projecting with the stored K and mapping the continuous
        coordinate the way the view moves the pixels (row offset, rescale with pixel centres, padding)."""
        pts = random_points()
        u, v, _ = project(self.Ks, pts)
        for mode, rows0, scale, pad in (("crop", [146], None, 0), ("split", [0, 292], None, 0),
                                        ("letterbox", [0], (297 / 518, 392 / 684), 110)):
            vs = self.views(mode)
            self.assertEqual(len(vs), len(rows0))
            for view, r0 in zip(vs, rows0):
                Kv = view.K(self.Ks, 518)
                uv, vv, _ = project(Kv, pts)
                ue, ve = u, v - r0
                if scale is not None:
                    ue, ve = (ue + 0.5) * scale[0] - 0.5, (ve + 0.5) * scale[1] - 0.5
                np.testing.assert_allclose(uv, ue + pad, atol=1e-9)
                np.testing.assert_allclose(vv, ve, atol=1e-9)
                self.assertEqual(view.out_wh, (518, 392))

    def test_letterbox_content_where_k_says(self):
        """A bright block in the stored image lands, after the letterbox resize and padding, where the view
        K predicts; masks use nearest neighbour (stay binary) and padding is reported."""
        view = self.views("letterbox")[0]
        img = np.zeros((684, 518), np.float32)
        img[300:316, 200:216] = 1.0  # centroid at pixel-centre coordinates (307.5, 207.5)
        out = view.image(img)
        self.assertEqual(out.shape, (392, 518))
        yy, xx = np.mgrid[0:392, 0:518]
        mu, mv = (out * xx).sum() / out.sum(), (out * yy).sum() / out.sum()
        self.assertAlmostEqual(mu, (207.5 + 0.5) * 297 / 518 - 0.5 + 110, delta=0.02)
        self.assertAlmostEqual(mv, (307.5 + 0.5) * 392 / 684 - 0.5, delta=0.02)
        m = view.mask(img > 0.5)
        self.assertEqual(m.dtype, bool)
        pad = view.padding()
        self.assertEqual(int(pad[0].sum()), 518 - 297)
        self.assertTrue(pad[:, :110].all() and pad[:, 407:].all() and not pad[:, 110:407].any())
        self.assertTrue(view.mask(np.zeros((684, 518), bool), pad_value=True)[:, :110].all())

    def test_sparse_depth_follows_view(self):
        """Sparse points move to the view pixel of their continuous position; nearest point kept."""
        view = self.views("letterbox")[0]
        vu = np.array([[300, 200], [301, 200], [600, 10]], np.uint16)
        z = np.array([5.0, 4.0, 30.0], np.float16)
        f = np.array([1, 2, 4], np.uint8)
        out_vu, out_z, out_f = view.sparse(vu, z, f, 518)
        # rows 300 and 301 map to the same view row (0.573 scale): the nearer point (4 m, flag 2) wins
        self.assertEqual(len(out_z), 2)
        r = int(np.floor((300 + 0.5) * 392 / 684 - 0.5 + 0.5))
        c = int(np.floor((200 + 0.5) * 297 / 518 - 0.5 + 110 + 0.5))
        i = [k for k in range(2) if tuple(out_vu[k]) == (r, c)]
        self.assertEqual(len(i), 1)
        self.assertEqual(float(out_z[i[0]]), 4.0)
        self.assertEqual(int(out_f[i[0]]), 2)
        crop = self.views("crop")[0]
        out_vu, out_z, _ = crop.sparse(vu, z, f, 518)
        self.assertEqual(out_vu.tolist(), [[154, 200], [155, 200]])  # row 600 is outside the crop


class TestRealCalibration(unittest.TestCase):
    def test_all_cameras(self):
        from realityview.data.av2_io import read_calibration
        from realityview.data.checks import check_intrinsics
        from realityview.data.config import CAMS
        from realityview.data.images import stored_intrinsics

        path, _, _ = _util.real_log()
        cfg = default_config()
        calib, _ = read_calibration(path)

        class Ctx:
            pass

        ctx = Ctx()
        ctx.calib = calib
        ctx.K = {c: stored_intrinsics(calib[c].K, (calib[c].width, calib[c].height), c, cfg) for c in CAMS}
        r = check_intrinsics(ctx, cfg)
        self.assertTrue(r["passed"], r)


if __name__ == "__main__":
    unittest.main()
