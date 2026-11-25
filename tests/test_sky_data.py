"""Sky scores in the processed-log format: storage, the reader, the check, and the padding used when scoring."""
import json
import os
import sys
import tempfile
import unittest

import _util  # noqa: F401
import cv2
import numpy as np
import torch
from test_dataset import N_FRAMES, build_fake_root

from realityview.data import dataset as dataset_module
from realityview.data.checks import check_sky
from realityview.data.config import CAMS, FRONT_CENTER, default_config
from realityview.data.dataset import Stage1Dataset
from realityview.data.images import stored_size
from realityview.data.store import SKY_DONE, LogStore, sky_relpath


def write_sky(root, done=True):
    """Sky scores for every image of the fake log: 1.2 in the top 50 rows, 0.4 in the next 20, 0 below."""
    cfg = default_config()
    log = os.path.join(root, "logs", "val", "fake")
    frames = LogStore(log).parquet("frames.parquet")
    for cam, ts in zip(frames["cam"], frames["timestamp_ns"]):
        w, h = stored_size(cam, cfg)
        m = np.zeros((h, w), np.uint8)
        m[:50] = 120
        m[50:70] = 40
        path = os.path.join(log, sky_relpath(cam, int(ts)))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, m)
    if done:
        with open(os.path.join(log, SKY_DONE), "w") as f:
            json.dump({"images": len(frames)}, f)
    return log, frames


class TestSkyInReader(unittest.TestCase):
    def test_reader_returns_raw_scores(self):
        with tempfile.TemporaryDirectory() as root:
            build_fake_root(root)
            write_sky(root)
            sample = Stage1Dataset(root, "val", ["fake"], with_sky=True, require_checks=False)[0]
            sky = sample["sky"]
            self.assertEqual((tuple(sky.shape), sky.dtype), ((14, 392, 518), torch.float32))
            landscape = sky[1]
            self.assertTrue(torch.allclose(landscape[:50], torch.tensor(1.2)))
            self.assertTrue(torch.allclose(landscape[50:70], torch.tensor(0.4)))
            self.assertEqual(float(landscape[70:].abs().max()), 0.0)
            # the front-center view is rows 146..538 of the stored picture: all of the sky band is cut away
            self.assertEqual(int(sample["cam_id"][0]), CAMS.index(FRONT_CENTER))
            self.assertEqual(float(sky[0].abs().max()), 0.0)

    def test_front_center_modes(self):
        with tempfile.TemporaryDirectory() as root:
            build_fake_root(root)
            write_sky(root)
            split = Stage1Dataset(root, "val", ["fake"], with_sky=True, require_checks=False, front_center_mode="split")[0]
            self.assertEqual(tuple(split["sky"].shape), (16, 392, 518))
            top = split["sky"][0]                       # rows 0..392 of the stored picture
            self.assertTrue(torch.allclose(top[:50], torch.tensor(1.2)))
            self.assertEqual(float(split["sky"][1].abs().max()), 0.0)   # rows 292..684
            letterbox = Stage1Dataset(root, "val", ["fake"], with_sky=True, require_checks=False,
                                      front_center_mode="letterbox")[0]
            view = letterbox["sky"][0]
            padded = letterbox["ego_mask"][0][0]        # the padded columns are marked in the ego mask
            self.assertEqual(float(view[:, padded].abs().max()), 0.0)
            self.assertAlmostEqual(float(view[5, 259]), 1.2, places=5)     # 50 rows of sky shrink to about 28
            self.assertEqual(float(view[60:, 259].abs().max()), 0.0)

    def test_a_log_without_sky_scores_is_skipped_with_a_reason(self):
        with tempfile.TemporaryDirectory() as root:
            build_fake_root(root)
            write_sky(root, done=False)
            ds = Stage1Dataset(root, "val", ["fake"], with_sky=True, require_checks=False)
            self.assertEqual(len(ds), 0)
            self.assertIn("sky", ds.skipped["fake"])
            self.assertEqual(len(Stage1Dataset(root, "val", ["fake"], require_checks=False).n_moments), 1)

    def test_storage_scale_is_the_same_on_both_sides(self):
        from realityview.models.sky import SKY_UINT8_SCALE

        self.assertEqual(dataset_module.SKY_UINT8_SCALE, SKY_UINT8_SCALE)


class TestSkyCheck(unittest.TestCase):
    def test_check(self):
        with tempfile.TemporaryDirectory() as root:
            build_fake_root(root)
            cfg = default_config()
            log = os.path.join(root, "logs", "val", "fake")
            frames = LogStore(log).parquet("frames.parquet")
            self.assertIsNone(check_sky(LogStore(log), frames, cfg)["passed"])     # not computed yet
            write_sky(root)
            result = check_sky(LogStore(log), frames, cfg)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["n_images"], 7 * N_FRAMES)
            self.assertGreater(result["sky_share_top_eighth"], 0.5)
            self.assertEqual(result["sky_share_bottom_eighth"], 0.0)
            cam, ts = frames["cam"].iloc[3], int(frames["timestamp_ns"].iloc[3])
            os.remove(os.path.join(log, sky_relpath(cam, ts)))
            broken = check_sky(LogStore(log), frames, cfg)
            self.assertFalse(broken["passed"])
            self.assertEqual(broken["n_missing"], 1)

    def test_sky_at_the_bottom_of_the_pictures_fails(self):
        with tempfile.TemporaryDirectory() as root:
            build_fake_root(root)
            log, frames = write_sky(root)
            for cam, ts in zip(frames["cam"], frames["timestamp_ns"]):
                path = os.path.join(log, sky_relpath(cam, int(ts)))
                m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                m[-30:] = 150
                cv2.imwrite(path, m)
            self.assertFalse(check_sky(LogStore(log), frames, default_config())["passed"])


class TestScoringHelpers(unittest.TestCase):
    def test_padding_to_the_patch_size(self):
        sys.path.insert(0, os.path.join(_util.REPO, "scripts"))
        try:
            from sky_scores import pad_to_patch
        finally:
            sys.path.pop(0)
        images = torch.arange(2 * 3 * 684 * 518, dtype=torch.float32).reshape(2, 3, 684, 518).remainder(251).to(torch.uint8)
        padded = pad_to_patch(images)
        self.assertEqual((tuple(padded.shape), padded.dtype), ((2, 3, 686, 518), torch.uint8))
        self.assertTrue(torch.equal(padded[:, :, :684], images))
        self.assertTrue(torch.equal(padded[:, :, 684], images[:, :, 683]))
        self.assertTrue(torch.equal(padded[:, :, 685], images[:, :, 683]))
        same = torch.zeros(1, 3, 392, 518, dtype=torch.uint8)
        self.assertIs(pad_to_patch(same), same)


class TestTaskHelpers(unittest.TestCase):
    def test_fixed_moments_and_log_lists(self):
        from realityview.tasks.stage1_backbone import fixed_moments, read_logs

        with tempfile.TemporaryDirectory() as root:
            build_fake_root(root)
            ds = Stage1Dataset(root, "val", ["fake"], require_checks=False)
            chosen = fixed_moments(ds, 4)
            self.assertEqual(chosen, sorted(set(chosen)))
            self.assertTrue(1 <= len(chosen) <= 4)
            self.assertEqual(fixed_moments(ds, 4), chosen)             # the same every time
            listing = os.path.join(root, "list.txt")
            with open(listing, "w") as f:
                f.write("# comment\naaa\nbbb extra\n\nccc\n")
            self.assertEqual(read_logs(listing), ["aaa", "bbb", "ccc"])
            self.assertEqual(read_logs(listing, first=2), ["aaa", "bbb"])
            self.assertEqual(read_logs(["x", "y"]), ["x", "y"])
            self.assertEqual(len(read_logs("data_lists/av2_train_stage1.txt")), 150)

    def test_losses_on_a_reader_sample(self):
        from realityview.tasks.stage1_backbone import Stage1BackboneTask

        with tempfile.TemporaryDirectory() as root:
            build_fake_root(root)
            write_sky(root)
            ds = Stage1Dataset(root, "val", ["fake"], with_sky=True, require_checks=False)
            batch = torch.utils.data.default_collate([ds[0]])
            task = Stage1BackboneTask({"model": {"weights_root": "unused"}, "data": {"root": root},
                                       "loss": {"anchor": {"min_points": 50}}})
            log_z = torch.full(batch["depth"].shape, float(np.log(20.0)), requires_grad=True)
            out = {"log_depth": log_z, "ref_log_depth": torch.full(batch["depth"].shape, float(np.log(10.0)))}
            total, terms, classes = task.losses(out, batch, step=0)
            self.assertEqual(set(terms), {"lidar", "sky", "anchor"})
            self.assertTrue(all(torch.isfinite(v) for v in terms.values()))
            self.assertGreater(float(classes["sky"][0, 1, :50].float().mean()), 0.7)   # the sky band, except near LiDAR points
            self.assertFalse(classes["sky"][0, 1, 50:].any())          # nothing below it, uncertain rows included
            has_point = batch["depth"][0, 1] > 0
            self.assertFalse((classes["sky"][0, 1] & has_point).any())
            self.assertFalse((classes["lidar"] & classes["sky"]).any())
            total.backward()
            self.assertTrue(torch.isfinite(log_z.grad).all())


if __name__ == "__main__":
    unittest.main()
