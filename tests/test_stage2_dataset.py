"""The stage-2 samples read from real data. Needs, besides the variables of _util.py:

    RV_CHASE_ROOT   output root of scripts/render_targets.py that contains RV_TEST_LOG
    RV_VIDEO_ROOT   output root of scripts/make_video_frames.py that contains RV_TEST_LOG

Skipped when they are not set.
"""
import json
import os
import unittest

import _util
import cv2
import numpy as np

from realityview.data.config import CAMS, FRONT_CENTER
from realityview.data.geometry import se3_inv

SIZE = (576, 320)
FRAMES = 17


def roots():
    out = {}
    for name in ("RV_CHASE_ROOT", "RV_VIDEO_ROOT"):
        if not os.environ.get(name):
            raise unittest.SkipTest(f"{name} not set")
        out[name] = os.environ[name]
    split = os.environ.get("RV_TEST_SPLIT", "val")
    log_id = os.environ.get("RV_TEST_LOG")
    if not log_id:
        raise unittest.SkipTest("RV_TEST_LOG not set")
    for root in out.values():
        if not os.path.isdir(os.path.join(root, split, log_id)):
            raise unittest.SkipTest(f"{split}/{log_id} not found under {root}")
    return _util.processed_root(), out["RV_CHASE_ROOT"], out["RV_VIDEO_ROOT"], split, log_id


class Stage2DatasetRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from realityview.data.stage2_dataset import Stage2Dataset

        root, cls.chase_root, cls.video_root, cls.split, cls.log_id = roots()
        cls.ds = Stage2Dataset(root, cls.split, [cls.log_id], chase_root=cls.chase_root, video_root=cls.video_root,
                               clip_frames=FRAMES, cond_every_frames=8, size=SIZE)
        if not cls.ds.starts:
            raise unittest.SkipTest(f"no clips: {cls.ds.skipped}")
        cls.start = len(cls.ds.starts) // 2
        cls.n_var = len(cls.ds.variants)

    def sample(self, variant):
        return self.ds[self.start * self.n_var + self.ds.variants.index(variant)]

    def test_counts(self):
        self.assertEqual(self.n_var, 2 * len(CAMS))
        self.assertEqual(len(self.ds), len(self.ds.starts) * self.n_var)
        self.assertEqual(self.ds.skipped, {})

    def test_shapes(self):
        s = self.sample(-1)
        M, N = 3, 2 * len(CAMS)                   # 17 frames = 5 latent frames, one moment per 2 latent frames
        F = (FRAMES - 1) // 4 + 1
        self.assertEqual(tuple(s["images"].shape[:3]), (M, N, 3))
        self.assertEqual(tuple(s["target"].shape), (FRAMES, 3, SIZE[1], SIZE[0]))
        self.assertEqual(str(s["target"].dtype), "torch.uint8")
        self.assertEqual(tuple(s["conf"].shape), (FRAMES, SIZE[1] // 8, SIZE[0] // 8))
        self.assertEqual(tuple(s["supervise"].shape), (SIZE[1] // 8, SIZE[0] // 8))
        self.assertEqual(tuple(s["tgt_clip_T_cam"].shape), (F, 4, 4))
        for key in ("K", "ref_T_cam", "clip_T_cam"):
            self.assertEqual(tuple(s[key].shape[:2]), (M, N), key)
        for key, value in s.items():
            if hasattr(value, "dtype") and value.dtype.is_floating_point:
                self.assertTrue(bool(value.isfinite().all()), key)

    def test_chase_target(self):
        from realityview.data.stage2_dataset import fit_image, fit_intrinsics

        s = self.sample(-1)
        folder = os.path.join(self.chase_root, self.split, self.log_id)
        with open(os.path.join(folder, "camera.json")) as f:
            cam = json.load(f)
        self.assertEqual(int(s["kind"]), 0)
        self.assertTrue(bool(s["view_valid"].all()))
        np.testing.assert_allclose(s["tgt_ego_T_cam"].numpy(), np.array(cam["ego_T_cam"]), atol=1e-5)
        K = fit_intrinsics(np.array(cam["K"], np.float64), (cam["width"], cam["height"]), SIZE)
        np.testing.assert_allclose(s["tgt_K"].numpy(), K, rtol=1e-5)
        li, i0 = self.ds.starts[self.start]
        t = self.ds.chase[li]["ts"][i0]
        img = fit_image(cv2.imread(os.path.join(folder, "rgb", f"{t}.jpg")), SIZE)[:, :, ::-1]
        np.testing.assert_array_equal(s["target"][0].permute(1, 2, 0).numpy(), img)
        conf = s["conf"].numpy()
        self.assertTrue(conf.min() >= 0.0 and conf.max() <= 1.0)
        self.assertGreater(conf.mean(), 0.01)
        sup = s["supervise"].numpy()
        self.assertTrue(sup.min() >= 0.0 and sup.max() == 1.0)
        self.assertLess(sup.min(), 1.0, "the ego vehicle should hide part of the chase picture")
        # the chase camera is fixed to the vehicle: seen from the front-center camera of the same frame it
        # is where camera.json says
        L = self.ds._log(li)
        expect = se3_inv(np.asarray(L["ego_T_cam"][FRONT_CENTER], np.float64)) @ np.array(cam["ego_T_cam"])
        np.testing.assert_allclose(s["tgt_clip_T_cam"][0].numpy(), expect, atol=2e-3)

    def test_real_target_removes_its_camera(self):
        cam_i = list(CAMS).index(FRONT_CENTER)
        s = self.sample(cam_i)
        self.assertEqual(int(s["kind"]), 1)
        self.assertEqual(int(s["tgt_cam"]), cam_i)
        valid = s["view_valid"].numpy()
        self.assertEqual(int((~valid).sum()), 2)                          # at t and at t - 0.5 s
        cam_id = s["cam_id"].numpy()
        self.assertTrue((cam_id[:, ~valid] == cam_i).all())
        self.assertEqual(int(s["images"][:, ~valid].abs().sum()), 0)
        self.assertGreater(int(s["images"][:, valid].abs().sum()), 0)
        self.assertEqual(float(s["conf"].min()), 1.0)
        li, i0 = self.ds.starts[self.start]
        L = self.ds._log(li)
        t = int(L["moments"].iloc[i0][f"ts_{FRONT_CENTER}"])
        img = cv2.imread(os.path.join(self.video_root, self.split, self.log_id, FRONT_CENTER, f"{t}.jpg"))
        from realityview.data.stage2_dataset import fit_image

        np.testing.assert_array_equal(s["target"][0].permute(1, 2, 0).numpy(), fit_image(img, SIZE)[:, :, ::-1])
        # the clip reference is this very camera at the first frame
        np.testing.assert_allclose(s["tgt_clip_T_cam"][0].numpy(), np.eye(4), atol=1e-4)
        # later frames: the vehicle has not jumped (under 40 m/s)
        shift = np.linalg.norm(s["tgt_clip_T_cam"][-1, :3, 3].numpy())
        self.assertLess(shift, 40.0 * (FRAMES - 1) / 20.0)

    def test_same_inputs_for_both_kinds(self):
        a, b = self.sample(-1), self.sample(0)
        valid = b["view_valid"].numpy()
        np.testing.assert_array_equal(a["images"][:, valid].numpy(), b["images"][:, valid].numpy())
        np.testing.assert_array_equal(a["ref_T_cam"].numpy(), b["ref_T_cam"].numpy())


if __name__ == "__main__":
    unittest.main()
