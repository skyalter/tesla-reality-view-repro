"""Models: the DA3 backbone and depth head for stage 1, and DA3METRIC sky scores.

- ``backbone.DA3Backbone``   multi-view DA3 ViT with known poses, time encoding, freezing, checkpointing
- ``depth_head.DepthHead``   DualDPT main branch (depth, confidence, per-pixel features)
- ``stage1.Stage1Model``     backbone + head -> metric depth, with a frozen reference branch
- ``sky.SkyScorer``          DA3METRIC-LARGE raw sky scores
- ``cameras``                image normalisation and the pose convention

Weights are read from ``$RV_WEIGHTS_ROOT/<model name>/model.safetensors`` unless a path is given.
"""
