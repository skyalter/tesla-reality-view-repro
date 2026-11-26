"""Stage 2 on the CPU with a tiny network: the diffusion backbone, the conditions, the schedule, the loss and
one training step of the whole model (with a stand-in for the stage-1 network and for the video autoencoder)."""
import math
import unittest

import _util  # noqa: F401
import numpy as np
import torch
import torch.nn as nn

from realityview.data import stage2_dataset as SD
from realityview.losses import diffusion as D
from realityview.models.diffusion import condition as C
from realityview.models.diffusion import schedule
from realityview.models.diffusion.model import Stage2Model, build_dit
from realityview.models.diffusion.toy_vae import ToyVAE
from realityview.models.diffusion.wan_dit import CONFIGS

H, W, PATCH_HW = 392, 518, (28, 37)


class FakeStage1(nn.Module):
    """Has the interface of Stage1Model and remembers what it was given."""

    class _Backbone(nn.Module):
        embed_dim, out_layers = 8, [0, 1, 2, 3]

    def __init__(self):
        super().__init__()
        self.backbone = self._Backbone()
        self.w = nn.Parameter(torch.ones(1))
        self.calls = []

    def forward(self, images, ref_T_cam, K, dt, with_reference=False, return_backbone_features=False):
        self.calls.append(images.clone())
        M, N = images.shape[:2]
        g = torch.Generator().manual_seed(int(images.float().sum()) % 1000)
        feats = [torch.randn(M, N, PATCH_HW[0] * PATCH_HW[1], 16, generator=g) for _ in range(4)]
        return {"depth": torch.full((M, N, H, W), 10.0), "backbone_feats": feats, "patch_hw": PATCH_HW}


def batch_of(B=2, M=2, N=14, T=9, hw=(32, 64), drop_cam=None):
    g = torch.Generator().manual_seed(0)
    eye = torch.eye(4)
    b = {
        "images": torch.randint(1, 255, (B, M, N, 3, H, W), generator=g, dtype=torch.uint8),
        "K": torch.tensor([[300.0, 0, 258.5], [0, 300.0, 195.5], [0, 0, 1]]).expand(B, M, N, 3, 3).clone(),
        "ref_T_cam": eye.expand(B, M, N, 4, 4).clone(), "clip_T_cam": eye.expand(B, M, N, 4, 4).clone(),
        "dt": torch.zeros(B, M, N), "src_time": torch.zeros(B, M, N),
        "cam_id": (torch.arange(N) % 7).expand(B, M, N).clone(), "slot": (torch.arange(N) // 7).expand(B, M, N).clone(),
        "view_valid": torch.ones(B, N, dtype=torch.bool),
        "target": torch.randint(0, 255, (B, T, 3, *hw), generator=g, dtype=torch.uint8),
        "tgt_K": torch.tensor([[40.0, 0, 31.5], [0, 40.0, 15.5], [0, 0, 1]]).expand(B, 3, 3).clone(),
        "tgt_ego_T_cam": eye.expand(B, 4, 4).clone(),
        "tgt_clip_T_cam": eye.expand(B, (T - 1) // 4 + 1, 4, 4).clone(),
        "tgt_time": torch.zeros(B, (T - 1) // 4 + 1),
    }
    if drop_cam is not None:
        b["view_valid"][0] = b["cam_id"][0, 0] != drop_cam
    return b


def tiny_model(**kw):
    torch.manual_seed(0)
    cfg = CONFIGS["tiny"]
    return Stage2Model(FakeStage1(), build_dit("tiny", None, kw.pop("grad_checkpoint", False)), ToyVAE(),
                       torch.randn(3, cfg["text_dim"]), amp_dtype=None, **kw)


class TestWanDiT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.dit = build_dit("tiny", None, False)
        nn.init.normal_(self.dit.head.head.weight, std=0.2)          # the official init leaves the output at zero
        self.x = torch.randn(2, 16, 3, 4, 8)
        self.t = torch.tensor([100.0, 900.0])
        self.text = torch.randn(2, 5, CONFIGS["tiny"]["text_dim"])
        self.tokens = torch.randn(2, 3, 11, CONFIGS["tiny"]["dim"])

    def test_new_attention_starts_without_effect(self):
        a = self.dit(self.x, self.t, self.text)
        b = self.dit(self.x, self.t, self.text, self.tokens)
        self.assertEqual(a.shape, self.x.shape)
        self.assertGreater(float(a.abs().mean()), 1e-3)
        self.assertTrue(torch.equal(a, b))
        for blk in self.dit.blocks:
            nn.init.normal_(blk.feat_attn.o.weight, std=0.2)
        self.assertGreater(float((self.dit(self.x, self.t, self.text, self.tokens) - a).abs().mean()), 1e-4)

    def test_feature_tokens_act_per_latent_frame(self):
        for blk in self.dit.blocks[-1:]:
            nn.init.normal_(blk.feat_attn.o.weight, std=0.2)
        for blk in self.dit.blocks:                                   # no mixing between frames after the feature step
            nn.init.zeros_(blk.self_attn.o.weight), nn.init.zeros_(blk.self_attn.o.bias)
        a = self.dit(self.x, self.t, self.text, self.tokens)
        tokens = self.tokens.clone()
        tokens[:, 2] += 1.0
        diff = (self.dit(self.x, self.t, self.text, tokens) - a).abs().amax(dim=(0, 1, 3, 4))
        self.assertEqual(float(diff[0]), 0.0)
        self.assertEqual(float(diff[1]), 0.0)
        self.assertGreater(float(diff[2]), 1e-5)

    def test_masked_tokens_do_not_count(self):
        for blk in self.dit.blocks:
            nn.init.normal_(blk.feat_attn.o.weight, std=0.2)
        mask = torch.ones(2, 11, dtype=torch.bool)
        mask[0, 7:] = False
        a = self.dit(self.x, self.t, self.text, self.tokens, feat_mask=mask)
        tokens = self.tokens.clone()
        tokens[0, :, 7:] = 50.0
        b = self.dit(self.x, self.t, self.text, tokens, feat_mask=mask)
        self.assertLess(float((a - b).abs().max()), 1e-5)
        self.assertGreater(float((self.dit(self.x, self.t, self.text, tokens) - a).abs().max()), 1e-4)

    def test_gradient_checkpointing_gives_the_same_gradients(self):
        for blk in self.dit.blocks:
            nn.init.normal_(blk.feat_attn.o.weight, std=0.2)
        grads = []
        for on in (False, True):
            self.dit.grad_checkpoint = on
            self.dit.train()
            self.dit.zero_grad()
            self.dit(self.x, self.t, self.text, self.tokens).pow(2).mean().backward()
            grads.append(torch.cat([p.grad.flatten() for p in self.dit.parameters() if p.grad is not None]))
        self.assertLess(float((grads[0] - grads[1]).abs().max()), 1e-6)

    def test_extra_is_added_to_the_patch_embedding(self):
        a = self.dit(self.x, self.t, self.text)
        extra = torch.zeros(2, 3 * 2 * 4, CONFIGS["tiny"]["dim"])
        self.assertTrue(torch.equal(a, self.dit(self.x, self.t, self.text, extra=extra)))
        self.assertGreater(float((self.dit(self.x, self.t, self.text, extra=torch.randn_like(extra)) - a).abs().mean()), 1e-4)


class TestConditions(unittest.TestCase):
    def test_token_centres_and_depth(self):
        uv = C.token_centres((H, W), PATCH_HW)
        self.assertEqual(tuple(uv.shape), (14 * 19, 2))
        self.assertAlmostEqual(float(uv[0, 0]), 13.5, places=4)
        self.assertAlmostEqual(float(uv[0, 1]), 13.5, places=4)
        self.assertAlmostEqual(float(uv[18, 0]), (504 + 517) / 2, places=4)      # the last column is cut by the image
        depth = torch.full((1, 2, H, W), 7.0)
        self.assertTrue(torch.allclose(C.token_log_depth(depth, PATCH_HW), torch.full((1, 2, 266), math.log(7.0)), atol=1e-5))

    def test_pooled_features(self):
        f = torch.zeros(1, 2, 28 * 37, 3)
        f[0, 0] = torch.arange(28 * 37.0)[:, None]
        p = C.pool_features([f, 2 * f], PATCH_HW)
        self.assertEqual(tuple(p.shape), (1, 2, 266, 6))
        self.assertAlmostEqual(float(p[0, 0, 0, 0]), (0 + 1 + 37 + 38) / 4, places=4)
        self.assertAlmostEqual(float(p[0, 0, 18, 0]), (36 + 73) / 2, places=4)      # the last column has one patch
        self.assertAlmostEqual(float(p[0, 0, 0, 3]), 2 * float(p[0, 0, 0, 0]), places=4)
        cond = C.Conditioner(3, 2, 48)
        cam = torch.zeros(1, 2, dtype=torch.long)
        self.assertEqual(tuple(cond.feature_tokens(p, cam, cam).shape), (1, 2 * 266, 48))

    def test_landing_pixel_matches_a_direct_projection(self):
        torch.manual_seed(0)
        M, N, F = 2, 3, 3
        K = torch.tensor([[310.0, 0, 250.0], [0, 305.0, 190.0], [0, 0, 1]]).expand(M, N, 3, 3)
        ang = torch.tensor(0.2)
        R = torch.tensor([[torch.cos(ang), 0, torch.sin(ang)], [0, 1, 0], [-torch.sin(ang), 0, torch.cos(ang)]])
        clip_T_cam = torch.eye(4).repeat(M, N, 1, 1)
        clip_T_cam[1, :, :3, :3] = R
        clip_T_cam[1, :, :3, 3] = torch.tensor([0.5, 0.1, 1.0])
        tgt = torch.eye(4).repeat(F, 1, 1)
        tgt[:, :3, 3] = torch.tensor([[0.0, -1.0, -6.0], [0.2, -1.0, -5.0], [0.4, -1.0, -4.0]])
        tgt_K = torch.tensor([[400.0, 0, 415.5], [0, 400.0, 239.5], [0, 0, 1]])
        logd = torch.full((M, N, 266), math.log(12.0))
        mof = torch.tensor([0, 0, 1])
        g = C.token_geometry(logd, K, clip_T_cam, torch.zeros(M, N), tgt, tgt_K, (480, 832), torch.tensor([0.0, 0.2, 0.4]),
                             mof, (H, W), PATCH_HW)
        self.assertEqual(tuple(g.shape), (F, N * 266, C.GEOM_DIM))
        # token 100 of view 1 for latent frame 2 (moment 1), by hand
        uv = C.token_centres((H, W), PATCH_HW)[100]
        p_cam = torch.tensor([(uv[0] - 250.0) / 310.0, (uv[1] - 190.0) / 305.0, 1.0]) * 12.0
        p_clip = clip_T_cam[1, 1, :3, :3] @ p_cam + clip_T_cam[1, 1, :3, 3]
        p_tgt = p_clip - tgt[2, :3, 3]
        un = (400.0 * p_tgt[0] / p_tgt[2] + 415.5) / 831 * 2 - 1
        vn = (400.0 * p_tgt[1] / p_tgt[2] + 239.5) / 479 * 2 - 1
        off = 6 + 3 * (1 + 2 * C.POINT_FREQS)
        got = g[2, 1 * 266 + 100]
        self.assertAlmostEqual(float(got[off]) * 2.0, float(un), places=4)
        self.assertAlmostEqual(float(got[off + 1]) * 2.0, float(vn), places=4)
        self.assertAlmostEqual(float(got[6]) * 4.0 * C.GEOM_SCALE_M, float(p_tgt[0]), places=3)
        flags = off + 2 * (1 + 2 * C.PIXEL_FREQS)
        self.assertAlmostEqual(float(got[flags]) * 5.0, math.log(float(p_tgt[2])), places=4)
        # without the depth head only the ray and the time remain
        g0 = C.token_geometry(logd, K, clip_T_cam, torch.zeros(M, N), tgt, tgt_K, (480, 832), torch.tensor([0.0, 0.2, 0.4]),
                              mof, (H, W), PATCH_HW, use_depth=False)
        self.assertTrue(torch.equal(g0[..., :6], g[..., :6]))
        self.assertEqual(float(g0[..., 6:flags + 3].abs().max()), 0.0)
        self.assertAlmostEqual(float(g[2, 0, -(1 + 2 * C.TIME_FREQS)]), -0.4, places=5)

    def test_target_rays(self):
        K = torch.tensor([[[400.0, 0, 415.5], [0, 400.0, 239.5], [0, 0, 1]]])
        ego_T_cam = torch.eye(4)[None].clone()
        ego_T_cam[0, :3, 3] = torch.tensor([-13.0, 0.0, 9.0])
        r = C.target_rays(K, ego_T_cam, (480, 832))
        self.assertEqual(tuple(r.shape), (1, 30 * 52, C.TARGET_RAY_DIM))
        self.assertTrue(torch.allclose(r[0, :, :3].norm(dim=-1), torch.ones(30 * 52), atol=1e-5))
        centre = r[0].reshape(30, 52, -1)[14:16, 25:27, :3].mean(dim=(0, 1))
        self.assertLess(float((centre / centre.norm() - torch.tensor([0.0, 0.0, 1.0])).abs().max()), 1e-3)
        cond = C.Conditioner(16, 4, 48)
        self.assertEqual(float(cond.target_embedding(r, 3).detach().abs().max()), 0.0)


class TestScheduleAndLoss(unittest.TestCase):
    def test_levels(self):
        sig, w = schedule.training_levels(5.0)
        self.assertEqual(sig.numel(), 1000)
        self.assertAlmostEqual(float(sig[0]), 1.0, places=6)
        self.assertAlmostEqual(float(sig[500]), 5 * 0.5 / (1 + 4 * 0.5), places=5)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=4)
        self.assertEqual(float(w.min()), 0.0)
        s = schedule.sampling_levels(10, 5.0)
        self.assertEqual((float(s[0]), float(s[-1])), (1.0, 0.0))

    def test_the_sampler_follows_an_exact_field(self):
        x0 = torch.randn(1, 2, 3)
        noise = torch.randn(1, 2, 3)
        out = schedule.sample_euler(lambda x, t: noise - x0, x0.shape, steps=7, noise=noise)
        self.assertLess(float((out - x0).abs().max()), 1e-5)
        sigma = torch.tensor([0.3])
        xt = schedule.add_noise(x0, noise, sigma)
        self.assertTrue(torch.allclose(xt, 0.7 * x0 + 0.3 * noise))

    def test_pooling_and_weights(self):
        x = torch.arange(9.0).view(1, 9, 1, 1).expand(1, 9, 2, 2)
        self.assertEqual(D.temporal_pool(x)[0, :, 0, 0].tolist(), [0.0, 2.5, 6.5])
        conf = torch.tensor([0.0, 0.25, 0.5, 1.0]).view(1, 1, 1, 4)
        w = D.confidence_weight(conf, torch.tensor([1.0, 1.0, 1.0, 0.0]).view(1, 1, 1, 4), w_min=0.1, c_ref=0.5)
        self.assertTrue(torch.allclose(w.flatten(), torch.tensor([0.1, 0.55, 1.0, 0.0])))
        pred = torch.randn(1, 3, 1, 1, 4, requires_grad=True)
        loss, per = D.weighted_flow_loss(pred, torch.zeros_like(pred), w)
        loss.backward()
        self.assertEqual(float(pred.grad[..., 3].abs().max()), 0.0)        # no supervision on the ego vehicle
        self.assertGreater(float(pred.grad[..., 0].abs().max()), 0.0)
        expect = (pred.detach().pow(2).mean(dim=1) * w).sum() / w.sum()
        self.assertAlmostEqual(float(per[0]), float(expect), places=6)

    def test_soft_confidence_is_the_renderer_formula(self):
        try:
            from realityview.render.targets import soft_confidence
        except Exception as error:  # noqa: BLE001
            raise unittest.SkipTest(f"renderer not importable: {error!r}")
        rng = np.random.default_rng(0)
        a, v, d = (rng.random((5, 7)).astype(np.float32) for _ in range(3))
        ref = soft_confidence(torch.from_numpy(a), torch.from_numpy(v), torch.from_numpy(d)).numpy()
        self.assertLess(float(np.abs(SD.soft_confidence(a, v, d) - ref).max()), 1e-6)

    def test_fitting_an_image_keeps_the_geometry(self):
        K = np.array([[900.0, 0, 639.5], [0, 900.0, 359.5], [0, 0, 1]])
        Kf = SD.fit_intrinsics(K, (1280, 720), (832, 480))
        (w, h), (c0, r0) = SD.fit_size((1280, 720), (832, 480))
        self.assertEqual((h, r0), (480, 0))
        p = np.array([0.7, -0.2, 5.0])
        u, v = K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]
        uf, vf = Kf[0, 0] * p[0] / p[2] + Kf[0, 2], Kf[1, 1] * p[1] / p[2] + Kf[1, 2]
        self.assertAlmostEqual(uf, (u + 0.5) * w / 1280 - 0.5 - c0, places=6)
        self.assertAlmostEqual(vf, (v + 0.5) * h / 720 - 0.5 - r0, places=6)
        img = np.zeros((720, 1280, 3), np.uint8)
        self.assertEqual(SD.fit_image(img, (832, 480)).shape, (480, 832, 3))


class TestStage2Model(unittest.TestCase):
    def test_a_training_step(self):
        model = tiny_model()
        batch = batch_of()
        out = model(batch, torch.tensor([0.4, 0.8]), None, torch.tensor([False, True]))
        self.assertEqual(tuple(out["pred"].shape), (2, 16, 3, 4, 8))
        loss, _ = D.weighted_flow_loss(out["pred"], out["target"], torch.ones(2, 3, 4, 8))
        loss.backward()
        groups = model.param_groups()
        self.assertTrue(all(p.grad is not None for p in groups["new"]))
        self.assertTrue(all(p.grad is not None for p in groups["pretrained"]))
        self.assertTrue(all(".self_attn." in n for n, p in model.dit.named_parameters()
                            if p.requires_grad and ".feat_attn." not in n and ".norm_feat." not in n))
        self.assertFalse(any(p.requires_grad for p in model.stage1.parameters()))
        frozen = model.frozen_prefixes()
        state = [k for k in model.state_dict() if not k.startswith(frozen)]
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(set(state), trainable)

    def test_scopes(self):
        n = {}
        for scope in ("adapters", "adapters+self_attn", "full"):
            model = tiny_model(scope=scope)
            n[scope] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertLess(n["adapters"], n["adapters+self_attn"])
        self.assertLess(n["adapters+self_attn"], n["full"])

    def test_a_removed_camera_never_reaches_the_backbone(self):
        model = tiny_model()
        batch = batch_of(drop_cam=3)
        batch["images"][0, :, batch["cam_id"][0, 0] == 3] = 77          # would be visible if it got through
        out = model(batch, torch.tensor([0.5, 0.5]))
        calls = model.stage1.calls
        self.assertEqual(calls[0].shape[1], 12)
        self.assertEqual(calls[1].shape[1], 14)
        self.assertFalse(bool((calls[0] == 77).all(dim=(0, 2, 3, 4)).any()))
        self.assertTrue(torch.isfinite(out["pred"]).all())

    def test_sampling_shapes(self):
        model = tiny_model().eval()
        batch = batch_of(B=1)
        lat = model.sample(batch, 9, (32, 64), steps=2, guidance=2.0, generator=torch.Generator().manual_seed(0))
        self.assertEqual(tuple(lat.shape), (1, 16, 3, 4, 8))
        self.assertEqual(tuple(model.decode(lat).shape), (1, 9, 3, 32, 64))

    def test_moment_of_frame(self):
        model = tiny_model(cond_every=2)
        self.assertEqual(model.moment_of_frame(11, 6, "cpu").tolist(), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])


if __name__ == "__main__":
    unittest.main()
