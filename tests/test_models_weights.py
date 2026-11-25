"""Strict loading of the three DA3 weight files (needs RV_WEIGHTS_ROOT)."""
from __future__ import annotations

import os
import unittest

from _models_common import weights, weights_root

from realityview.models.da3 import build as B


class TestWeights(unittest.TestCase):
    def test_config_json_matches(self):
        root = weights_root()
        for name in ("da3-base", "da3-large-1.1", "da3metric-large"):
            B.check_config(name, os.path.join(root, name, "config.json"))

    def test_multiview_models_load_strictly(self):
        for name, n_file in (("da3-base", 437), ("da3-large-1.1", 637)):
            with self.subTest(name=name):
                sd, aliases = B.read_weights(weights(name))
                self.assertEqual(len(sd), n_file)
                # 6 LayerNorm tensors of the auxiliary branch are aliases without data of their own
                self.assertEqual(len(aliases), 6)
                self.assertTrue(all("output_conv2_aux" in k for k in aliases))
                bb, ce, cd, head_sd, rep = B.load_backbone_parts(name, weights(name), with_cam_dec=True)
                # backbone + cam_enc + cam_dec + head main branch used, the 94 auxiliary tensors dropped
                self.assertEqual(rep["used"] + rep["dropped"], n_file)
                self.assertEqual(rep["dropped"], 94)
                from realityview.models.depth_head import DepthHead
                head = DepthHead.from_model_name(name)
                B.load_strict(head, head_sd)
                n_main = len(head.state_dict())
                n_bb, n_ce, n_cd = len(bb.state_dict()), len(ce.state_dict()), len(cd.state_dict())
                self.assertEqual(n_bb + n_ce + n_cd + n_main + 94, n_file)
                print(f"\n[weights] {name}: file {n_file} tensors = backbone {n_bb} + cam_enc {n_ce} + "
                      f"cam_dec {n_cd} (dropped in our model) + head main {n_main} + head aux 94 (dropped)",
                      flush=True)
                # the original model with the auxiliary branch also loads strictly once aliases are filled
                B.DA3Original(name, weights(name), with_cam_dec=True)

    def test_metric_model_loads_strictly(self):
        from realityview.models.sky import SkyScorer
        s = SkyScorer(weights("da3metric-large"))
        self.assertEqual(s.load_report, {"file": 406, "used": 406, "dropped": 0})

    def test_missing_or_extra_tensor_is_an_error(self):
        sd, _ = B.read_weights(weights("da3-base"))
        bb = B.build_backbone("da3-base")
        part = B.component(sd, "backbone")
        part.pop(next(iter(part)))
        with self.assertRaises(RuntimeError):
            B.load_strict(bb, part)
        sd2 = dict(sd)
        sd2["something_else.weight"] = sd[next(iter(sd))]
        with self.assertRaises(ValueError):
            B.account(sd2, ["backbone", "cam_enc", "head"], ["cam_dec.", "_aux"])


if __name__ == "__main__":
    unittest.main()
