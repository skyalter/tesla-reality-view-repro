"""Pose convention: our ref_T_cam (camera -> reference, OpenCV axes, metres) against DA3's camera encoder."""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch
from _models_common import load_av2_sample, need_cuda, to_batch, weights

from realityview.models.cameras import da3_extrinsics, da3_pose_encoding, normalize_images


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _quat_xyzw(R):
    """Independent rotation-matrix -> quaternion (x, y, z, w), w >= 0 (Shepperd's method)."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return -q if q[3] < 0 else q


class TestPoseEncoding(unittest.TestCase):
    def test_encoding_of_known_poses(self):
        """The 9 numbers DA3 encodes are [camera centre / scale in the reference frame, c2w rotation as
        quaternion x y z w (scalar last), fov_h, fov_w], for rotated example cameras."""
        H, W, scale = 392, 518, 20.0
        poses, Ks = [], []
        rots = [np.eye(3), _rot("y", 90), _rot("y", -45) @ _rot("x", 10), _rot("z", 30) @ _rot("y", 170)]
        centres = [np.zeros(3), [1.5, 0.2, -0.4], [-3.0, 0.0, 12.0], [0.3, -2.0, -8.0]]
        for R, c in zip(rots, centres):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, c
            poses.append(T)
            Ks.append(np.array([[300.0, 0, 259], [0, 310.0, 196], [0, 0, 1]]))
        ref_T_cam = torch.tensor(np.stack(poses), dtype=torch.float32)[None]
        K = torch.tensor(np.stack(Ks), dtype=torch.float32)[None]
        enc = da3_pose_encoding(ref_T_cam, K, (H, W), pose_scale=scale)[0].double().numpy()
        for i, (R, c) in enumerate(zip(rots, centres)):
            np.testing.assert_allclose(enc[i, :3], np.asarray(c) / scale, atol=1e-6)
            np.testing.assert_allclose(enc[i, 3:7], _quat_xyzw(R), atol=1e-5)
            self.assertAlmostEqual(enc[i, 7], 2 * math.atan(H / 2 / 310.0), places=5)
            self.assertAlmostEqual(enc[i, 8], 2 * math.atan(W / 2 / 300.0), places=5)
        # the reference camera: zero translation, identity quaternion (0, 0, 0, 1)
        np.testing.assert_allclose(enc[0, :7], [0, 0, 0, 0, 0, 0, 1], atol=1e-7)

    def test_extrinsics_are_world_to_camera(self):
        """da3_extrinsics(ref_T_cam) maps reference-frame points into the camera: x_cam = E @ x_ref."""
        R, c = _rot("y", 30) @ _rot("x", -5), np.array([2.0, -1.0, 6.0])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, c
        E = da3_extrinsics(torch.tensor(T, dtype=torch.float32)[None, None], pose_scale=1.0)[0, 0].double().numpy()
        p_cam = np.array([0.5, -0.2, 4.0, 1.0])
        p_ref = T @ p_cam
        np.testing.assert_allclose(E @ p_ref, p_cam, atol=1e-5)
        # scale divides the translation only
        E20 = da3_extrinsics(torch.tensor(T, dtype=torch.float32)[None, None], pose_scale=20.0)[0, 0].double()
        np.testing.assert_allclose(E20.numpy()[:3, :3], R.T, atol=1e-6)
        np.testing.assert_allclose(E20.numpy()[:3, 3], -R.T @ c / 20.0, atol=1e-6)

    def test_normalize_images(self):
        u8 = torch.randint(0, 256, (2, 3, 4, 5), dtype=torch.uint8)
        torch.testing.assert_close(normalize_images(u8), normalize_images(u8.float() / 255))


class TestPoseConventionOnRealImages(unittest.TestCase):
    def test_da3_predicts_our_poses(self):
        """Run DA3 without poses and decode its own camera estimate (cam_dec): the relative rotations agree
        with our ref_T_cam to a few degrees, while reading ref_T_cam the other way round (as
        world-to-camera) disagrees by up to ~160 degrees. This pins camera axes and direction of the poses
        against what DA3 learned, independently of the algebra above."""
        dev = need_cuda()
        from realityview.models.da3.build import DA3Original

        m = DA3Original("da3-large-1.1", weights("da3-large-1.1"), with_cam_dec=True).to(dev)
        for frame in (100, 200):
            s = load_av2_sample(frame=frame)
            imgs, K, T, dt = to_batch(s, dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                r = m(normalize_images(imgs), ref_view_strategy="first")
            w2c = r["extrinsics"][0].double().cpu().numpy()
            w2c = np.concatenate([w2c, np.tile([[[0, 0, 0, 1.0]]], (len(w2c), 1, 1))], 1)
            pred = np.stack([w2c[0] @ np.linalg.inv(w) for w in w2c])  # camera i -> camera 0
            ours = s[2].astype(np.float64)
            swapped = np.linalg.inv(ours)

            def rot_err(A, Bm):
                return [math.degrees(math.acos(np.clip((np.trace(A[i, :3, :3].T @ Bm[i, :3, :3]) - 1) / 2, -1, 1)))
                        for i in range(1, len(A))]

            e_ours, e_swap = rot_err(pred, ours), rot_err(pred, swapped)
            print(f"\n[pose convention] frame {frame}: rotation error ours max {max(e_ours):.1f} mean "
                  f"{np.mean(e_ours):.1f} deg; swapped max {max(e_swap):.1f} mean {np.mean(e_swap):.1f} deg",
                  flush=True)
            self.assertLess(max(e_ours), 15.0)
            self.assertGreater(np.mean(e_swap), 45.0)


if __name__ == "__main__":
    unittest.main()
