"""Stage-1 model: equivalence with the original DA3, view order, time encoding, freezing, frozen reference.

Equivalence and view-order tests need RV_WEIGHTS_ROOT, AV2_RAW_ROOT and CUDA; the parameter counts need
nothing.
"""
from __future__ import annotations

import unittest

import torch
from _models_common import load_av2_sample, need_cuda, rel_diff, to_batch, weights

from realityview.models import backbone as backbone_mod
from realityview.models.cameras import da3_extrinsics, normalize_images
from realityview.models.da3 import build as B
from realityview.models.da3.dinov2 import vision_transformer as vt
from realityview.models.stage1 import Stage1Model

LARGE, BASE = "da3-large-1.1", "da3-base"


def _n(params):
    return sum(p.numel() for p in params)


class TestEquivalenceWithOriginal(unittest.TestCase):
    """Stage1Model (time encoding 0, kappa 1) against DA3 assembled from the copied original code, on one real
    AV2 sample: 7 cameras x {t, t - 0.5 s}, 518 x 392, poses given (bfloat16 autocast like the official API)."""

    @classmethod
    def setUpClass(cls):
        cls.dev = need_cuda()
        cls.sample = load_av2_sample(frame=100)

    def _compare(self, name):
        dev = self.dev
        imgs, K, T, dt = to_batch(self.sample, dev)
        orig = B.DA3Original(name, weights(name)).to(dev)
        # world-to-camera extrinsics computed in float32 outside autocast, as the official API does
        # (``api.py`` normalises the extrinsics before entering ``forward``)
        ext = da3_extrinsics(T, 20.0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            r = orig(normalize_images(imgs), ext, K)
        del orig
        results = {}
        for ckpt in (False, True):
            m = Stage1Model(name, weights(name), grad_checkpoint=ckpt, head_checkpoint=ckpt).to(dev)
            with torch.no_grad():
                o = m(imgs, T, K, dt)
                if not ckpt:  # a training loop may wrap the call in autocast: nothing may change
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        results["outer_autocast"] = m(imgs, T, K, dt)
            results[ckpt] = o
            del m
        o = results[False]
        dmax, dmean = rel_diff(o["depth"] / 20.0, r["depth"])
        print(f"\n[equivalence] {name}: metric depth / 20 vs original depth: rel diff max {dmax:.2e} mean "
              f"{dmean:.2e}; raw exp(l) identical: {torch.equal(o['rel_depth'], r['depth'])}; conf identical: "
              f"{torch.equal(o['depth_conf'], r['depth_conf'])}; frozen reference identical: "
              f"{torch.equal(o['ref_depth'], o['depth'])}; checkpointing on/off identical: "
              f"{torch.equal(results[True]['depth'], o['depth'])}; inside an outer autocast identical: "
              f"{torch.equal(results['outer_autocast']['depth'], o['depth'])}", flush=True)
        self.assertLess(dmax, 1e-5)
        self.assertTrue(torch.equal(o["rel_depth"], r["depth"]))
        self.assertTrue(torch.equal(o["depth_conf"], r["depth_conf"]))
        self.assertTrue(torch.equal(o["ref_depth"], o["depth"]))
        self.assertTrue(torch.equal(results[True]["depth"], o["depth"]))
        self.assertTrue(torch.equal(results["outer_autocast"]["depth"], o["depth"]))

    def test_large(self):
        self._compare(LARGE)

    def test_base(self):
        self._compare(BASE)


class TestViewOrder(unittest.TestCase):
    """With poses DA3 never reorders the views; without poses it does (and restores the order)."""

    @classmethod
    def setUpClass(cls):
        cls.dev = need_cuda()
        cls.sample = load_av2_sample(frame=100)
        cls.bb = backbone_mod.DA3Backbone(LARGE, weights(LARGE)).to(cls.dev)

    def test_no_reordering_with_poses(self):
        calls = []
        orig_select = vt.select_reference_view

        def spy(x, strategy="saddle_balanced"):
            idx = orig_select(x, strategy=strategy)
            calls.append(idx.tolist())
            return idx

        vt.select_reference_view = spy
        try:
            imgs, K, T, dt = to_batch(self.sample, self.dev)
            with torch.no_grad():
                self.bb(imgs, T, K, dt)
                self.assertEqual(calls, [])  # the pose path never selects a reference view
                self.bb.forward_without_poses(imgs)
            self.assertEqual(len(calls), 1)
            print(f"\n[view order] with poses: no reference-view selection; without poses DA3 selected view "
                  f"{calls[0]} as reference (and restores the order on output)", flush=True)
        finally:
            vt.select_reference_view = orig_select

    def test_permuting_views_permutes_outputs(self):
        """Output view i belongs to input view i: feeding the views in another order gives the same per-view
        features, up to bfloat16 summation order."""
        imgs, K, T, dt = to_batch(self.sample, self.dev)
        perm = torch.tensor([7, 3, 12, 0, 9, 1, 13, 5, 2, 11, 4, 6, 10, 8], device=self.dev)
        with torch.no_grad():
            a = self.bb(imgs, T, K, dt)["feats"][-1]
            b = self.bb(imgs[:, perm], T[:, perm], K[:, perm], dt[:, perm])["feats"][-1]
        diff = (b - a[:, perm]).norm(dim=-1) / a[:, perm].norm(dim=-1)
        wrong = (b - a).norm(dim=-1) / a.norm(dim=-1)
        print(f"\n[view order] last-layer features, permuted input vs permuted output: relative L2 per token "
              f"mean {diff.mean():.2e} max {diff.max():.2e}; vs unpermuted output mean {wrong.mean():.2e}",
              flush=True)
        self.assertLess(float(diff.mean()), 0.02)
        self.assertGreater(float(wrong.mean()), 0.3)


class TestTimeEncoding(unittest.TestCase):
    def test_zero_init_is_exact_and_nonzero_changes_output(self):
        dev = need_cuda()
        s = load_av2_sample(frame=100)
        imgs, K, T, dt = to_batch(s, dev)
        bb = backbone_mod.DA3Backbone(BASE, weights(BASE)).to(dev)
        self.assertEqual(float(bb.time_enc.proj.weight.detach().abs().sum() + bb.time_enc.proj.bias.detach().abs().sum()), 0.0)
        with torch.no_grad():
            without = bb(imgs, T, K, None)["feats"]
            with_dt = bb(imgs, T, K, dt)["feats"]
            for a, b in zip(without, with_dt):
                self.assertTrue(torch.equal(a, b))
            torch.nn.init.normal_(bb.time_enc.proj.weight, std=0.05)
            changed = bb(imgs, T, K, dt)["feats"]
        d = (changed[-1] - without[-1]).abs().mean() / without[-1].abs().mean()
        print(f"\n[time encoding] zero init: bit-identical; random weights (std 0.05): mean relative change of "
              f"the last feature layer {float(d):.2e}", flush=True)
        self.assertGreater(float(d), 1e-3)


class TestFreezing(unittest.TestCase):
    def test_trainable_parameter_counts(self):
        """Counts per preset (random init on CPU, no weights needed)."""
        for name in (LARGE, BASE):
            m = Stage1Model(name, load_weights=False, frozen_reference=False)
            bb = m.backbone
            total = _n(m.parameters())
            lines = []
            for preset in ("default", "late", "frozen"):
                bb.set_freeze(preset)
                g = m.param_groups()
                nb = _n(g["backbone"])
                blocks = [i for i, blk in enumerate(bb.vit.blocks) if any(p.requires_grad for p in blk.parameters())]
                lines.append(f"{preset}: blocks {blocks[0] if blocks else '-'}..{blocks[-1] if blocks else '-'} "
                             f"backbone+cam_enc+time {nb / 1e6:.1f}M, head {_n(g['head']) / 1e6:.1f}M")
                per_block = _n(bb.vit.blocks[-1].parameters())
                if preset == "default":
                    self.assertEqual(blocks, list(range(bb.alt_start, bb.depth)))
                    self.assertEqual(nb, sum(_n(bb.vit.blocks[i].parameters()) for i in blocks)
                                     + _n(bb.cam_enc.parameters()) + _n(bb.time_enc.parameters()))
                elif preset == "late":
                    self.assertEqual(blocks, list(range(bb.depth - bb.depth // 4, bb.depth)))
                    self.assertEqual(nb, per_block * len(blocks))
                else:
                    self.assertEqual(nb, 0)
                # always frozen
                for p in (bb.vit.patch_embed, bb.vit.norm):
                    self.assertFalse(any(q.requires_grad for q in p.parameters()))
                self.assertFalse(bb.vit.pos_embed.requires_grad or bb.vit.cls_token.requires_grad)
            print(f"\n[freeze] {name}: total {total / 1e6:.1f}M parameters; " + "; ".join(lines), flush=True)
        big = Stage1Model(LARGE, load_weights=False, frozen_reference=False)
        self.assertEqual(big.backbone.trainable_blocks, list(range(8, 24)))
        big.backbone.set_freeze("late")
        self.assertEqual(big.backbone.trainable_blocks, list(range(18, 24)))

    def test_only_trainable_parts_get_gradients(self):
        dev = need_cuda()
        m = Stage1Model(LARGE, weights(LARGE), frozen_reference=False, grad_checkpoint=True).to(dev)
        torch.nn.init.normal_(m.backbone.time_enc.proj.weight, std=1e-3)  # so the time path carries gradient
        g = torch.Generator().manual_seed(0)
        imgs = torch.randint(0, 256, (1, 3, 3, 196, 266), generator=g, dtype=torch.uint8).to(dev)
        T = torch.eye(4).repeat(1, 3, 1, 1).to(dev)
        T[0, 1, 0, 3], T[0, 2, 2, 3] = 0.5, -5.0
        K = torch.tensor([[150.0, 0, 133], [0, 150.0, 98], [0, 0, 1]]).repeat(1, 3, 1, 1).to(dev)
        dt = torch.tensor([[0.0, 0.0, -0.5]], device=dev)
        for preset in ("default", "late", "frozen"):
            m.backbone.set_freeze(preset)
            m.zero_grad(set_to_none=True)
            out = m(imgs, T, K, dt)
            (out["log_depth"].mean() + out["depth_conf"].mean() * 1e-3).backward()
            has = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
            want = {n for n, p in m.named_parameters() if p.requires_grad}
            # LayerScale-free parameters with a structurally zero gradient do not exist here; demand equality
            missing = sorted(want - has)
            extra = sorted({n for n, p in m.named_parameters() if p.grad is not None} - want)
            print(f"\n[freeze] {preset}: {len(want)} trainable tensors, {len(has)} with nonzero gradient, "
                  f"missing {missing[:5]}, frozen with gradient {extra[:5]}", flush=True)
            self.assertEqual(extra, [])
            self.assertEqual(missing, [])


class TestFrozenReference(unittest.TestCase):
    def test_reference_fixed_while_training(self):
        dev = need_cuda()
        s = load_av2_sample(frame=150, offsets=(0,))
        imgs, K, T, dt = to_batch(s, dev)
        imgs, K, T, dt = imgs[:, :3], K[:, :3], T[:, :3], dt[:, :3]
        for preset in ("default", "late"):
            m = Stage1Model(BASE, weights(BASE), freeze=preset, grad_checkpoint=True).to(dev)
            n_ref = _n(m.reference.parameters())
            self.assertFalse(any(p.requires_grad for p in m.reference.parameters()))
            with torch.no_grad():
                o0 = m(imgs, T, K, dt)
            self.assertTrue(torch.equal(o0["ref_depth"], o0["depth"]))
            opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
            for _ in range(3):
                o = m(imgs, T, K, dt)
                loss = (o["log_depth"] - torch.log(torch.full_like(o["log_depth"], 10.0))).abs().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                o1 = m(imgs, T, K, dt)
            moved = rel_diff(o1["depth"], o0["depth"])[1]
            print(f"\n[frozen reference] da3-base preset {preset}: reference starts at layer "
                  f"{m.reference.start}, copies {n_ref / 1e6:.1f}M parameters; initial reference == trainable "
                  f"depth; after 3 AdamW steps trainable depth moved by mean {moved:.2e}, reference identical: "
                  f"{torch.equal(o1['ref_depth'], o0['ref_depth'])}", flush=True)
            self.assertGreater(moved, 1e-3)
            self.assertTrue(torch.equal(o1["ref_depth"], o0["ref_depth"]))
            del m, opt


if __name__ == "__main__":
    unittest.main()
