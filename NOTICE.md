# Third-party material

## Ego-vehicle masks for Argoverse 2

`assets/ego_masks/argoverse2/ring_front_center.png`, `ring_rear_left.png`, `ring_rear_right.png`

Copied unchanged from [drivestudio](https://github.com/ziyc/drivestudio) (commit `e59bda4`), files
`data/ego_masks/argoverse/0.png`, `5.png` and `6.png`, renamed after the cameras they belong to.
White pixels are the ego vehicle's body at the original image resolution.

MIT License, Copyright (c) 2024 Ziyu Chen. The full license text is in `assets/ego_masks/LICENSE.drivestudio`.

## Log lists for Argoverse 2

`data_lists/av2_*_log_city.tsv` were produced by listing the public Argoverse 2 S3 bucket; the city code
comes from the name of each log's map file. Argoverse 2 is licensed CC BY-NC-SA 4.0, (c) Argo AI, LLC.

## Depth Anything 3 model code

`realityview/models/da3/` (except `build.py` and the package `__init__.py` files, which are ours)

Copied from [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (commit `3d835ec`),
`src/depth_anything_3/model/`: the DINOv2 backbone as modified by DA3 (`dinov2/`), the reference-view
selector, the camera encoder and decoder, the DPT and DualDPT heads and their helpers (`utils/`).
`utils/geometry.py` and `utils/constants.py` are trimmed copies of `src/depth_anything_3/utils/`. Changes:
relative imports, `addict.Dict` replaced by `dict`, no `xformers` import, standard `logging` with the
per-forward messages at DEBUG level. Each file keeps its original copyright header; changed files say so
under the header.

Apache License 2.0, Copyright (c) 2025 ByteDance Ltd. and/or its affiliates; the DINOv2 files are
Copyright (c) Meta Platforms, Inc. and affiliates. The full license text is in `realityview/models/da3/LICENSE`.

The pretrained weights are not part of this repository. DA3-BASE is published under Apache-2.0; for
DA3-LARGE-1.1 the Hugging Face model card says Apache-2.0 while the DA3 README says CC BY-NC 4.0.
