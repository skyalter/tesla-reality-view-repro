"""Our DepthHead (DualDPT main branch only) against the original DualDPT (needs RV_WEIGHTS_ROOT and CUDA)."""
from __future__ import annotations

import unittest

import torch
from _models_common import need_cuda, rel_diff, weights

from realityview.models.da3 import build as B
from realityview.models.depth_head import DepthHead


def _heads(name, dev):
    sd, aliases = B.read_weights(weights(name))
    orig = B.build_original_head(name)
    B.load_strict(orig, B.component(B.fill_aliases(sd, aliases), "head"))
    ours = DepthHead.from_model_name(name)
    B.load_strict(ours, B.component(sd, "head", drop=("_aux",)))
    return orig.to(dev).eval(), ours.to(dev).eval()


def _feats(name, b, n, h, w, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dim = B.model_config(name)["head"]["dim_in"]
    # token statistics roughly like the backbone's: first half un-normalised, second half normalised
    return [(torch.randn(b, n, h * w, dim, generator=g) * 3.0).to(dev) for _ in range(4)]


class TestDepthHead(unittest.TestCase):
    def setUp(self):
        self.dev = need_cuda()
        self._tf32 = torch.backends.cudnn.allow_tf32
        # compare float32 to float32: TF32 convolutions make results depend on the batch split
        torch.backends.cudnn.allow_tf32 = False

    def tearDown(self):
        torch.backends.cudnn.allow_tf32 = self._tf32

    def test_same_depth_as_dualdpt(self):
        for name in ("da3-base", "da3-large-1.1"):
            orig, ours = _heads(name, self.dev)
            for b, n in ((1, 14), (2, 5)):
                feats = _feats(name, b, n, 28, 37, self.dev)
                with torch.no_grad():
                    o = orig([(f, None) for f in feats], 392, 518, patch_start_idx=0)
                    for chunk in (8, None, 3):
                        m = ours(feats, (392, 518), chunk_size=chunk, return_features=True)
                        dmax, dmean = rel_diff(m["depth"], o["depth"])
                        cmax, cmean = rel_diff(m["depth_conf"], o["depth_conf"])
                        print(f"\n[head] {name} B={b} N={n} chunk={chunk}: depth rel diff max {dmax:.2e} "
                              f"mean {dmean:.2e}; conf max {cmax:.2e}", flush=True)
                        self.assertLess(dmax, 1e-3)
                        self.assertLess(cmax, 1e-3)
                        self.assertEqual(tuple(m["features"].shape),
                                         (b, n, ours.feature_dim, 392, 518))
                        torch.testing.assert_close(m["log_depth"].exp(), m["depth"])
                    if b == 1:  # DA3's own chunking (8): identical to the last bit
                        m = ours(feats, (392, 518), chunk_size=8)
                        self.assertTrue(torch.equal(m["depth"], o["depth"]))

    def test_features_feed_the_depth(self):
        """depth = exp(output_conv2(F)[0]): the returned features are the last stage's input."""
        orig, ours = _heads("da3-base", self.dev)
        feats = _feats("da3-base", 1, 2, 28, 37, self.dev)
        with torch.no_grad():
            m = ours(feats, (392, 518), return_features=True)
            f = m["features"].flatten(0, 1)
            l = ours.scratch.output_conv2(f)[:, 0].reshape(m["log_depth"].shape)
        torch.testing.assert_close(l, m["log_depth"])


if __name__ == "__main__":
    unittest.main()
