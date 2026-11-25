# tesla-reality-view-repro

An independent attempt to reproduce the **Reality View** of Tesla FSD v15 using public material only:
public statements about the system, public datasets and open-source code. Not affiliated with Tesla.

**Status: work in progress. Nothing is trained yet.** At this point the repository only holds the data
download tooling and the cluster helper scripts. Models, training and evaluation code will follow.

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

## Layout

```
scripts/      command-line tools (data download so far)
lighthouse/   helper scripts for a Slurm cluster
data_lists/   which Argoverse 2 logs are used for what
assets/       small third-party files needed at run time (see NOTICE.md)
```

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
