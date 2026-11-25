"""Depth losses and metrics on hand-made maps where the right answer is known."""
from __future__ import annotations

import math
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from realityview.losses import depth as D  # noqa: E402


def maps(h=12, w=16, b=1, n=2):
    lidar = torch.zeros(b, n, h, w)
    flag = torch.zeros(b, n, h, w, dtype=torch.uint8)
    ego = torch.zeros(b, n, h, w, dtype=torch.bool)
    sky = torch.zeros(b, n, h, w)
    return lidar, flag, ego, sky


class PixelClassesTest(unittest.TestCase):
    def test_each_rule(self):
        lidar, flag, ego, sky = maps()
        lidar[0, 0, 2, 2] = 10.0                       # plain LiDAR point
        lidar[0, 0, 2, 4] = 0.3                        # too near
        lidar[0, 0, 2, 6] = 230.0                      # beyond the range used for the loss
        lidar[0, 0, 2, 8] = 10.0
        flag[0, 0, 2, 8] = 4 | 1                       # suspected see-through
        lidar[0, 0, 2, 10] = 10.0
        ego[0, 0, 2, 10] = True                        # on the ego vehicle
        lidar[0, 0, 2, 12] = 10.0
        sky[0, 0, 2, 12] = 0.4                         # uncertain sky score
        flag[0, 0, 2, 2] = 2                           # other flags do not matter
        sky[0, 1, :4] = 0.9                            # sky band in the second view
        lidar[0, 1, 1, 3] = 50.0                       # a LiDAR point inside the sky band
        ego[0, 1, 0, 15] = True
        sky[0, 1, 4] = 0.4                             # the uncertain band below the sky
        c = D.pixel_classes(lidar, flag, ego, sky, sky_low=0.3, sky_high=0.5, sky_window=5)

        self.assertEqual(c["lidar"].nonzero().tolist(), [[0, 0, 2, 2]])
        # anchor: no point at all, not ego, clearly not sky
        self.assertTrue(c["anchor"][0, 0, 5, 5])
        for u in (2, 4, 6, 8, 10, 12):
            self.assertFalse(c["anchor"][0, 0, 2, u], u)
        self.assertFalse(c["anchor"][0, 1, 4].any())           # uncertain sky: in no class
        self.assertFalse(c["sky"][0, 1, 4].any())
        self.assertFalse(c["lidar"][0, 1].any())
        # sky: high score, not ego, and no LiDAR point within the 5x5 window
        self.assertTrue(c["sky"][0, 1, 0, 8])
        self.assertFalse(c["sky"][0, 1, 0, 15])
        self.assertFalse(c["sky"][0, 1, 3, 5])                 # 2 pixels from the point at (1, 3)
        self.assertTrue(c["sky"][0, 1, 3, 6])                  # 3 columns away: outside the window
        self.assertFalse(c["sky"][0, 0].any())
        self.assertFalse((c["lidar"] & c["anchor"]).any() or (c["sky"] & c["anchor"]).any())

    def test_without_sky_scores_nothing_is_sky(self):
        lidar, flag, ego, _ = maps()
        lidar[0, 0, 1, 1] = 5.0
        c = D.pixel_classes(lidar, flag, ego, None)
        self.assertFalse(c["sky"].any())
        self.assertEqual(int(c["lidar"].sum()), 1)
        self.assertEqual(int(c["anchor"].sum()), lidar.numel() - 1)


class LossTest(unittest.TestCase):
    def test_lidar_loss_value_and_gradient_stay_inside_the_mask(self):
        lidar, flag, ego, _ = maps()
        lidar[0, 0, 3, 3] = 10.0
        lidar[0, 1, 5, 7] = 40.0
        mask = D.pixel_classes(lidar, flag, ego)["lidar"]
        pred = torch.full_like(lidar, 20.0, requires_grad=True)
        loss = D.lidar_depth_loss(pred, lidar, mask)
        self.assertAlmostEqual(float(loss.detach()), math.log(2.0), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(int((pred.grad != 0).sum()), 2)
        self.assertAlmostEqual(float(D.lidar_depth_loss(lidar.clamp(min=1.0), lidar, mask)), 0.0, places=6)

    def test_samples_count_equally_and_empty_ones_are_skipped(self):
        lidar, flag, ego, _ = maps(b=3)
        lidar[0, 0, 0, :8] = 10.0                      # 8 points, prediction twice too far
        lidar[1, 0, 0, 0] = 10.0                       # 1 point, prediction exact
        mask = D.pixel_classes(lidar, flag, ego)["lidar"]
        pred = torch.full_like(lidar, 10.0)
        pred[0] = 20.0
        self.assertAlmostEqual(float(D.lidar_depth_loss(pred, lidar, mask)), math.log(2.0) / 2, places=6)
        nothing = torch.zeros_like(mask)
        empty = D.lidar_depth_loss(pred.requires_grad_(), lidar, nothing)
        self.assertEqual(float(empty), 0.0)
        empty.backward()                               # still connected to the graph

    def test_confidence_weighting(self):
        lidar, flag, ego, _ = maps()
        lidar[0, 0, 3, 3] = 10.0
        mask = D.pixel_classes(lidar, flag, ego)["lidar"]
        pred = torch.full_like(lidar, 20.0)
        conf = torch.full_like(lidar, 3.0)
        expected = 3.0 * math.log(2.0) - 0.2 * math.log(3.0)
        self.assertAlmostEqual(float(D.lidar_depth_loss(pred, lidar, mask, conf, 0.2)), expected, places=6)

    def test_ray_norms(self):
        K = torch.tensor([[100.0, 0.0, 8.0], [0.0, 50.0, 6.0], [0.0, 0.0, 1.0]]).expand(1, 2, 3, 3)
        norms = D.ray_norms(K, 12, 16)
        self.assertEqual(tuple(norms.shape), (1, 2, 12, 16))
        self.assertAlmostEqual(float(norms[0, 0, 6, 8]), 1.0, places=6)
        self.assertAlmostEqual(float(norms[0, 1, 0, 0]), math.sqrt(0.08 ** 2 + 0.12 ** 2 + 1.0), places=6)

    def test_sky_loss_is_about_distance_not_depth(self):
        K = torch.tensor([[10.0, 0.0, 8.0], [0.0, 10.0, 6.0], [0.0, 0.0, 1.0]]).expand(1, 2, 3, 3)
        norms = D.ray_norms(K, 12, 16)
        mask = torch.zeros(1, 2, 12, 16, dtype=torch.bool)
        mask[0, 0, 0] = True
        pred = 400.0 / norms
        self.assertAlmostEqual(float(D.sky_distance_loss(pred, norms, mask, 400.0)), 0.0, places=5)
        self.assertAlmostEqual(float(D.sky_distance_loss(pred / 4, norms, mask, 400.0)), math.log(4.0), places=5)

    def test_reference_is_scaled_to_lidar_before_anchoring(self):
        g = torch.Generator().manual_seed(0)
        truth = 5.0 + 60.0 * torch.rand(2, 3, 12, 16, generator=g)
        reference = truth / 3.0
        reference[1] = truth[1] / 0.5                  # every sample has its own scale
        lidar = torch.zeros_like(truth)
        lidar[..., ::2, ::2] = truth[..., ::2, ::2]
        lidar_mask = lidar > 0
        lidar[0, 0, 0, 0] = 200.0                      # an outlier does not move the median
        aligned, valid = D.align_reference(reference, lidar, lidar_mask, "sample", min_points=20, max_depth=80.0)
        self.assertTrue(valid.all())
        self.assertTrue(torch.allclose(aligned, truth, rtol=1e-4))
        anchor_mask = ~lidar_mask
        self.assertAlmostEqual(float(D.anchor_loss(truth, aligned, anchor_mask & valid)), 0.0, places=5)
        self.assertAlmostEqual(float(D.anchor_loss(truth * 1.5, aligned, anchor_mask & valid)), math.log(1.5), places=5)

    def test_per_view_alignment_and_views_without_enough_lidar(self):
        g = torch.Generator().manual_seed(1)
        truth = 5.0 + 60.0 * torch.rand(1, 3, 12, 16, generator=g)
        scales = torch.tensor([2.0, 0.25, 7.0]).reshape(1, 3, 1, 1)
        reference = truth / scales
        lidar = torch.zeros_like(truth)
        lidar[:, :2, ::2, ::2] = truth[:, :2, ::2, ::2]
        lidar[0, 2, 0, :5] = truth[0, 2, 0, :5]        # only 5 points in the third view
        aligned, valid = D.align_reference(reference, lidar, lidar > 0, "view", min_points=20)
        self.assertTrue(torch.allclose(aligned[0, :2], truth[0, :2], rtol=1e-4))
        self.assertTrue(valid[0, :2].all())
        self.assertFalse(valid[0, 2].any())
        self.assertTrue(torch.equal(aligned[0, 2], reference[0, 2]))
        same, ok = D.align_reference(reference, lidar, lidar > 0, "none")
        self.assertTrue(torch.equal(same, reference) and ok.all())

    def test_the_reference_gets_no_gradient(self):
        pred = torch.full((1, 1, 4, 4), 10.0, requires_grad=True)
        ref = torch.full((1, 1, 4, 4), 12.0, requires_grad=True)
        D.anchor_loss(pred, ref, torch.ones(1, 1, 4, 4, dtype=torch.bool)).backward()
        self.assertIsNone(ref.grad)
        self.assertTrue((pred.grad != 0).all())


class MetricsTest(unittest.TestCase):
    def test_numbers_bands_and_cameras(self):
        gt = torch.zeros(1, 2, 2, 4)
        pred = torch.ones(1, 2, 2, 4)
        gt[0, 0, 0] = torch.tensor([5.0, 15.0, 30.0, 100.0])
        pred[0, 0, 0] = torch.tensor([5.5, 15.0, 45.0, 50.0])
        gt[0, 1, 1, 0], pred[0, 1, 1, 0] = 8.0, 8.0
        metrics = D.DepthMetrics(["front", "rear"])
        metrics.update("single", pred, gt, gt > 0, torch.tensor([[0, 1]]))
        out = metrics.compute()
        self.assertEqual(out["pixels_single"], 5.0)
        self.assertAlmostEqual(out["absrel_single"], (0.1 + 0.0 + 0.5 + 0.5 + 0.0) / 5)
        self.assertAlmostEqual(out["delta1_single"], 3 / 5)
        expected = math.sqrt((math.log(1.1) ** 2 + math.log(1.5) ** 2 + math.log(2.0) ** 2) / 5)
        self.assertAlmostEqual(out["rmse_log_single"], expected)
        self.assertAlmostEqual(out["absrel_single/0-10m"], 0.05)
        self.assertAlmostEqual(out["absrel_single/20-40m"], 0.5)
        self.assertAlmostEqual(out["absrel_single/80-200m"], 0.5)
        self.assertNotIn("absrel_single/40-80m", out)          # no pixels there
        self.assertAlmostEqual(out["absrel_single/front"], 1.1 / 4)
        self.assertAlmostEqual(out["absrel_single/rear"], 0.0)
        self.assertEqual(sum(v for k, v in out.items() if k.startswith("pixels_single/") and k.endswith("m")), 5.0)
        metrics.update("single", pred, gt, gt > 0, torch.tensor([[0, 1]]))     # a second batch: same averages
        self.assertAlmostEqual(metrics.compute()["absrel_single"], out["absrel_single"])
        self.assertEqual(metrics.compute()["pixels_single"], 10.0)

    def test_median_scale(self):
        gt = torch.full((2, 1, 4, 4), 10.0)
        pred = torch.full((2, 1, 4, 4), 4.0)
        mask = torch.ones(2, 1, 4, 4, dtype=torch.bool)
        mask[1] = False
        self.assertEqual(D.median_scale(pred, gt, mask).tolist(), [2.5, 1.0])


class OverlapTest(unittest.TestCase):
    def scene(self):
        """Two cameras 0.5 m apart looking the same way at a wall 10 m away, plus one camera at another time."""
        h, w = 40, 60
        K = torch.tensor([[50.0, 0.0, 29.5], [0.0, 50.0, 19.5], [0.0, 0.0, 1.0]]).repeat(3, 1, 1)
        T = torch.eye(4).repeat(3, 1, 1)
        T[1, 0, 3] = 0.5
        z = torch.full((3, h, w), 10.0)
        return z, K, T, torch.ones(3, h, w, dtype=torch.bool), torch.tensor([0, 0, 1])

    def test_consistent_depths_agree(self):
        z, K, T, valid, slot = self.scene()
        median, within, count = D.overlap_consistency(z, K, T, valid, slot)
        self.assertLess(median, 1e-6)
        self.assertEqual(within, 1.0)
        self.assertGreater(count, 2 * 40 * 50)                 # most of both images; the third view is not compared

    def test_a_wrong_view_shows_up(self):
        z, K, T, valid, slot = self.scene()
        z[1] *= 1.1
        median, within, _ = D.overlap_consistency(z, K, T, valid, slot)
        self.assertAlmostEqual(median, 0.1, delta=0.011)
        self.assertEqual(within, 0.0)

    def test_no_overlap_gives_none(self):
        z, K, T, valid, slot = self.scene()
        self.assertIsNone(D.overlap_consistency(z, K, T, valid, torch.tensor([0, 1, 2])))


if __name__ == "__main__":
    unittest.main()
