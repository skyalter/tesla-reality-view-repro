"""Sky scores from DA3METRIC-LARGE on real AV2 images (needs RV_WEIGHTS_ROOT, AV2_RAW_ROOT and CUDA).

Set RV_TEST_OUT to a directory to also save image | score visualisations there.
"""
from __future__ import annotations

import os
import unittest

import numpy as np
import torch
from _models_common import CAMS, load_av2_sample, need_cuda, weights

from realityview.models.sky import SkyScorer, sky_score_from_uint8, sky_score_to_uint8


class TestSky(unittest.TestCase):
    def test_uint8_round_trip(self):
        s = np.array([0.0, 0.004, 0.3, 0.5, 1.0, 1.234, 3.0], np.float32)
        q = sky_score_to_uint8(torch.from_numpy(s))
        self.assertEqual(q.dtype, np.uint8)
        np.testing.assert_allclose(sky_score_from_uint8(q)[:6], s[:6], atol=0.005 + 1e-6)
        self.assertEqual(q[-1], 255)

    def test_sky_above_road(self):
        dev = need_cuda()
        scorer = SkyScorer(weights("da3metric-large")).to(dev)
        s = load_av2_sample(frame=100, offsets=(0,))
        imgs = torch.from_numpy(s[0])
        sky = scorer(imgs.to(dev)).cpu()
        self.assertEqual(tuple(sky.shape), (7, 392, 518))
        self.assertGreaterEqual(float(sky.min()), 0.0)
        lines = []
        for i, cam in enumerate(CAMS):
            top = float(sky[i, :40].mean())          # upper tenth of the image
            road = float(sky[i, -80:].mean())        # lower fifth: road (and ego body on the rear cameras)
            lines.append(f"{cam}: top {top:.3f} road {road:.3f}")
            self.assertLess(road, 0.05)
        tops = [float(sky[i, :40].mean()) for i in range(7)]
        # every camera here sees sky in its upper part
        self.assertGreater(min(tops), 0.3)
        print("\n[sky] mean raw score, upper 40 rows vs lower 80 rows: " + "; ".join(lines), flush=True)
        out = os.environ.get("RV_TEST_OUT")
        if out:
            import cv2

            os.makedirs(out, exist_ok=True)
            for i in (0, 1, 4):
                im = s[0][i].transpose(1, 2, 0)[:, :, ::-1]
                q = sky_score_to_uint8(sky[i])
                heat = cv2.applyColorMap(np.clip(q.astype(np.int32) * 2, 0, 255).astype(np.uint8),
                                         cv2.COLORMAP_VIRIDIS)
                blend = (0.5 * im + 0.5 * heat).astype(np.uint8)
                cv2.imwrite(os.path.join(out, f"sky_{CAMS[i]}.jpg"), np.concatenate([im, heat, blend], 1))


if __name__ == "__main__":
    unittest.main()
