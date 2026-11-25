# tesla-reality-view-repro

An independent attempt to reproduce the **Reality View** of Tesla FSD v15 using public material only:
public statements about the system, public datasets and open-source code. Not affiliated with Tesla.

**Status: work in progress. Nothing is trained yet.** The repository holds the code of stage 1: data
download and processing, the backbone and depth head with the depth losses, the Gaussian head with rendering
and the photo losses, a multi-GPU training loop and the evaluations. The diffusion model will follow.

## What is being built

Reality View turns the surround cameras of a car into a photorealistic view from a virtual camera behind
and above the car, in real time. The plan follows what has been said publicly about its structure:

1. **A shared vision encoder.** A pretrained multi-view backbone (Depth Anything 3), fine-tuned on driving
   data and then frozen.
2. **A feed-forward Gaussian model** on top of the encoder: one 3D Gaussian per pixel, rendered with gsplat.
   No public dataset has ground truth for the chase view, so this model renders the chase-view images that
   the next stage learns from.
3. **A pose-conditioned video diffusion model** that takes the encoder features and a camera pose and
   outputs the chase-view video directly. The first version uses one fixed chase camera.

Stage 1 is trained in two parts. First the backbone alone, on depth: LiDAR depth, sky pushed far away,
and, where there is no LiDAR, staying close to a frozen copy of the original network; no rendering. Then
the backbone is frozen and the Gaussian head is trained with rendering losses: every input pixel becomes a
Gaussian at its predicted depth, and the Gaussians are rendered into the cameras of the input moment and of
another moment up to a second away and compared with the photos.

## Layout

```
realityview/
  data/       offline processing of Argoverse 2 logs and the training reader   (data/README.md)
  models/     backbone, depth head, frozen reference, Gaussian head, sky scorer; da3/ holds the
              Depth Anything 3 model files
  render/     one Gaussian per pixel, rendering with gsplat, coverage, the chase camera
  losses/     depth losses and depth metrics; losses that compare renders with photos
  engine/     training loop: multi-GPU, exact resume, checkpoints, configuration
  tasks/      what can be trained: the model, data and losses of one training run
configs/      one YAML file per run; open design choices are switches in there
scripts/      command-line tools: download, preprocess, sky scores, data checks, train, evaluate
tests/        unit tests
lighthouse/   scripts for a Slurm cluster                                       (lighthouse/README.md)
data_lists/   which Argoverse 2 logs are used for what
assets/       small third-party files needed at run time (see NOTICE.md)
```

## Running

Nothing is installed; run from the repository root. The packages are listed in `lighthouse/requirements.txt`.

```bash
# process raw logs, add sky scores, check (see realityview/data/README.md)
python scripts/preprocess.py --raw-root <raw> --out-root <processed>/av2_v1 --split train --logs data_lists/av2_train_stage1.txt --workers 8
python scripts/sky_scores.py --out-root <processed>/av2_v1 --split train --weights <weights>/da3metric-large/model.safetensors
python scripts/check_data.py --out-root <processed>/av2_v1 --split train --overlays --require-sky

# depth errors of the original pretrained network (the baseline), then of a checkpoint
python scripts/eval_depth.py --config configs/stage1_backbone.yaml --set data.root=<processed>/av2_v1 model.weights_root=<weights>
python scripts/eval_depth.py --config configs/stage1_backbone.yaml --checkpoint <run>/last.pt --set ...

# train the backbone: one process per GPU; the same command continues a run that was stopped
torchrun --nproc_per_node=8 scripts/train.py --config configs/stage1_backbone.yaml \
    --set run_dir=<run> data.root=<processed>/av2_v1 model.weights_root=<weights>

# on one small GPU: the smaller backbone memorises two logs
python scripts/train.py --config configs/stage1_overfit_base.yaml --set run_dir=<run> data.root=... model.weights_root=...

# train the Gaussian head on the frozen backbone (needs gsplat and lpips)
torchrun --nproc_per_node=8 scripts/train.py --config configs/stage1_head.yaml \
    --set run_dir=<run> data.root=<processed>/av2_v1 model.weights_root=<weights> \
          model.backbone_checkpoint=<backbone run>/last.pt

# on one small GPU: the head on the smaller backbone memorises two logs
python scripts/train.py --config configs/stage1_head_overfit_base.yaml --set run_dir=<run> data.root=... \
    model.weights_root=... model.backbone_checkpoint=<overfit backbone run>/last.pt
```

The head training writes comparison images to `<run>/eval_images/`: photos on top, renders below, one
column per camera.

Tests use the standard library only: `python -m unittest discover -s tests`. Tests that need real data,
pretrained weights or a GPU are skipped unless `AV2_RAW_ROOT`, `RV_PROCESSED_ROOT` and `RV_WEIGHTS_ROOT` are set.

## Data

[Argoverse 2 Sensor](https://www.argoverse.org/av2.html), downloaded from its public S3 bucket.
The dataset is licensed CC BY-NC-SA 4.0 (non-commercial); anything trained on it inherits that limit.

```bash
# a small local set for development: validation logs, training-log metadata, pretrained weights
scripts/download_debug_data.sh --data-root <dir> --weights-root <dir> --dry-run
```

See `lighthouse/README.md` for the full download on a cluster.

## License

Not chosen yet. Third-party files keep their own licenses, listed in `NOTICE.md`.
