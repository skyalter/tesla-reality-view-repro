"""The Gaussian head: appearance branch, exposure correction, the losses that compare renders with photos,
and the whole model on the frozen backbone (needs a GPU and RV_WEIGHTS_ROOT with da3-base/)."""
from __future__ import annotations

import math
import os
import tempfile
import unittest

import torch
from _models_common import H, W, need_cuda, weights

from realityview.data.config import FLAG_CENTER_SWEEP, FLAG_SEETHROUGH, FLAG_SYNC_DYNAMIC
from realityview.losses import render as R
from realityview.models.gs_head import AppearanceBranch, ExposureCorrection, Stage1HeadModel
from realityview.models.stage1 import Stage1Model
from realityview.render.camera import project


class Parts(unittest.TestCase):
    def test_appearance_branch_starts_at_its_biases(self):
        torch.manual_seed(0)
        branch = AppearanceBranch(48)
        x = torch.randn(3, 48, 10, 14)
        out = branch(x, chunk_size=2)
        expected = torch.tensor([-2.0, -2, -2, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(tuple(out.shape), (3, 11, 10, 14))
        self.assertTrue(torch.equal(out, expected[None, :, None, None].expand_as(out)))
        # once it has weights, the result does not depend on how the views are chunked
        torch.nn.init.normal_(branch.net[-1].weight, std=0.1)
        self.assertTrue(torch.allclose(branch(x, chunk_size=2), branch(x, chunk_size=None), atol=1e-6))

    def test_exposure_starts_as_identity(self):
        torch.manual_seed(0)
        ex = ExposureCorrection(32)
        A, b = ex(torch.randn(2, 3, 10, 32))
        self.assertEqual((tuple(A.shape), tuple(b.shape)), ((2, 3, 3, 3), (2, 3, 3)))
        self.assertTrue(torch.equal(A, torch.eye(3).expand(2, 3, 3, 3)))
        self.assertTrue(torch.equal(b, torch.zeros(2, 3, 3)))
        rgb = torch.rand(6, 3, 4, 5)
        self.assertTrue(torch.allclose(ExposureCorrection.apply(rgb, A.reshape(6, 3, 3), b.reshape(6, 3)), rgb))
        A2 = torch.diag(torch.tensor([0.5, 1.0, 2.0]))[None].repeat(6, 1, 1)
        b2 = torch.tensor([0.1, 0.0, -0.1])[None].repeat(6, 1)
        out = ExposureCorrection.apply(rgb, A2, b2)
        self.assertTrue(torch.allclose(out[:, 0], rgb[:, 0] * 0.5 + 0.1))
        self.assertTrue(torch.allclose(out[:, 2], (rgb[:, 2] * 2.0 - 0.1).clamp(0, 1)))

    def test_photo_measures(self):
        pred = torch.zeros(2, 3, 4, 4)
        target = torch.zeros(2, 3, 4, 4)
        target[0, :, :2] = 0.5                           # 8 of the 32 pixels differ by 0.5
        everything = torch.ones(2, 4, 4, dtype=torch.bool)
        self.assertAlmostEqual(float(R.masked_mse(pred, target, everything)), 0.25 * 8 / 32, places=6)
        only_wrong = torch.zeros(2, 4, 4, dtype=torch.bool)
        only_wrong[0, :2] = True
        self.assertAlmostEqual(float(R.masked_mse(pred, target, only_wrong)), 0.25, places=6)
        self.assertEqual(float(R.masked_mse(pred, target, torch.zeros(2, 4, 4, dtype=torch.bool))), 0.0)
        self.assertAlmostEqual(R.psnr(0.01), 20.0, places=6)

    def test_rendered_depth_against_lidar(self):
        depth = torch.full((1, 2, 6), 10.0)
        alpha = torch.tensor([[[0.9, 0.9, 0.9, 0.9, 0.2, 0.9], [0.9] * 6]])
        lidar = torch.tensor([[[10.0, 20.0, 300.0, 10.0, 10.0, 10.0], [0.0, 10.0, 10.0, 10.0, 10.0, 10.0]]])
        flag = torch.zeros(1, 2, 6, dtype=torch.uint8)
        flag[0, 0, 3] = FLAG_SYNC_DYNAMIC                # a point on a moving object
        flag[0, 1, 1] = FLAG_SEETHROUGH
        flag[0, 1, 2] = FLAG_CENTER_SWEEP                # harmless
        region = torch.ones(1, 2, 6, dtype=torch.bool)
        region[0, 1, 5] = False
        mask = R.rendered_depth_mask(depth, alpha, lidar, flag, region, alpha_min=0.5, lidar_range=(0.5, 200.0))
        expected = torch.tensor([[[True, True, False, False, False, True], [False, False, True, True, True, False]]])
        self.assertTrue(torch.equal(mask, expected))
        loss = R.rendered_depth_loss(depth, lidar, mask)
        self.assertAlmostEqual(float(loss), math.log(2.0) / 6, places=5)     # one of the six pixels is off by 2x


def cameras(n_slots: int = 2):
    """Two cameras (front, rotated right) at n_slots moments 5 m apart: (ref_T_cam, K, dt, slot)."""
    K = torch.tensor([[427.0, 0, 258.5], [0, 427.0, 195.5], [0, 0, 1]])
    poses, dts, slots = [], [], []
    for s in range(n_slots):
        for yaw in (0.0, 50.0):
            a = math.radians(yaw)
            T = torch.eye(4)
            T[:3, :3] = torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            T[:3, 3] = torch.tensor([0.3 * s, 0.0, -5.0 * s])
            poses.append(T)
            dts.append(-0.5 * s)
            slots.append(s)
    n = len(poses)
    return torch.stack(poses)[None], K[None, None].repeat(1, n, 1, 1), torch.tensor(dts)[None], torch.tensor(slots)[None]


class Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = need_cuda()
        cls.weights = weights("da3-base")
        torch.manual_seed(0)
        cls.model = Stage1HeadModel("da3-base", cls.weights).to(cls.dev)
        T, K, dt, slot = cameras()
        cls.inputs = [x.to(cls.dev) for x in (torch.rand(1, 4, 3, H, W), T, K, dt)]
        cls.slot = slot

    def test_backbone_is_frozen_and_stays_in_evaluation_mode(self):
        m = self.model
        self.assertFalse(any(p.requires_grad for p in m.stage1.backbone.parameters()))
        self.assertFalse(m.stage1.log_kappa.requires_grad)
        m.train()
        self.assertFalse(m.stage1.backbone.training)
        self.assertTrue(m.appearance.training)
        groups = m.param_groups()
        self.assertEqual(set(groups), {"depth_head", "appearance", "exposure"})
        named = {id(p) for g in groups.values() for p in g}
        self.assertEqual(named, {id(p) for p in m.parameters() if p.requires_grad})
        m.eval()

    def test_untrained_gaussians_sit_on_the_predicted_depth(self):
        """The depth of every Gaussian in its own camera is the predicted depth of its pixel, it projects to
        the centre of that pixel, and the depth is the one the backbone model predicts."""
        images, T, K, dt = self.inputs
        keep = torch.ones(1, 4, H, W, dtype=torch.bool, device=self.dev)
        keep[:, :, :, :40] = False
        with torch.no_grad():
            out = self.model(images, T, K, dt, keep)
            plain = Stage1Model("da3-base", self.weights, freeze="frozen", frozen_reference=False).to(self.dev).eval()
            ref = plain(images, T, K, dt)
        self.assertTrue(torch.allclose(out["depth"], ref["depth"], rtol=1e-4, atol=1e-4))
        g = out["gaussians"][0]
        self.assertEqual(len(g), int(keep.sum()))
        v, u = torch.meshgrid(torch.arange(H, device=self.dev), torch.arange(W, device=self.dev), indexing="ij")
        for n in range(4):
            mine = g.view == n
            uv, z = project(g.means[mine].double(), T[0, n].double(), K[0, n].double())
            self.assertLess(float((z / out["depth"][0, n][keep[0, n]].double() - 1).abs().max()), 1e-4)
            self.assertLess(float((uv[:, 0] - u[keep[0, n]]).abs().max()), 0.02)
            self.assertLess(float((uv[:, 1] - v[keep[0, n]]).abs().max()), 0.02)
        self.assertTrue(torch.allclose(g.opacities, torch.full_like(g.opacities, 1 / (1 + math.e ** 2)), atol=1e-6))
        self.assertTrue(torch.allclose(out["background"], torch.full((3,), 0.5, device=self.dev)))
        A, b = out["exposure"]
        self.assertTrue(torch.equal(A, torch.eye(3, device=self.dev).expand(1, 4, 3, 3)))

    def test_every_trainable_part_gets_a_gradient(self):
        from realityview.render.rasterize import render

        images, T, K, dt = self.inputs
        keep = torch.ones(1, 4, H, W, dtype=torch.bool, device=self.dev)
        self.model.train()
        out = self.model(images, T, K, dt, keep)
        res = render(out["gaussians"][0], T[0, :2], K[0, :2], H, W, background=out["background"])
        A, b = out["exposure"]
        rgb = ExposureCorrection.apply(res["rgb"], A[0, :2], b[0, :2])
        loss = ((rgb - images[0, :2]) ** 2).mean() + out["log_depth"].mean() + res["depth"].mean() * 1e-3
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        for name, params in self.model.param_groups().items():
            self.assertTrue(all(p.grad is not None for p in params), name)
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in params), 0.0, name)
        self.assertTrue(all(p.grad is None for p in self.model.stage1.backbone.parameters()))
        self.model.zero_grad(set_to_none=True)
        self.model.eval()

    def test_backbone_checkpoint(self):
        trained = Stage1Model("da3-base", self.weights, frozen_reference=False)
        with torch.no_grad():
            trained.head.scratch.output_conv2[-1].bias += 0.25
            trained.log_kappa.fill_(math.log(0.7))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "last.pt")
            torch.save({"model": trained.state_dict(), "cfg": {"model": {"pose_scale": 20.0}}, "step": 7}, path)
            model = Stage1HeadModel("da3-base", self.weights, backbone_checkpoint=path)
            self.assertEqual(model.backbone_checkpoint["step"], 7)
            self.assertAlmostEqual(float(model.stage1.log_kappa), math.log(0.7), places=6)
            self.assertTrue(torch.equal(model.stage1.head.scratch.output_conv2[-1].bias,
                                        trained.head.scratch.output_conv2[-1].bias))
            self.assertFalse(model.stage1.log_kappa.requires_grad)
            with self.assertRaises(ValueError):          # a backbone trained with another pose scale
                Stage1HeadModel("da3-base", self.weights, backbone_checkpoint=path, pose_scale=10.0)


if __name__ == "__main__":
    unittest.main()
