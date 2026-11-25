"""Model code copied from Depth Anything 3 (ByteDance-Seed/Depth-Anything-3, commit 3d835ec, Apache-2.0).

Only the files needed to build the backbone, the camera encoder, the DualDPT depth head and the DPT head of
DA3METRIC are copied, from ``src/depth_anything_3/model/``. Every file keeps its original copyright header;
files that were changed say so in a ``Modified in realityview`` line under the header. The changes are kept
small so that the files can be compared with upstream:

- imports made relative to this package;
- ``addict.Dict`` replaced by ``dict``; ``xformers`` never imported; the package logger replaced by the
  standard ``logging`` module, with the per-forward messages at DEBUG level;
- ``utils/geometry.py`` and ``utils/constants.py`` trimmed to the one function / constant that is used.

``build.py`` is ours: it builds the networks by model name and loads the safetensors weights strictly,
without the official ``DepthAnything3`` API (which runs under ``torch.inference_mode``).
The full license text is in ``LICENSE`` next to this file.
"""
