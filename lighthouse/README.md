# Running on a Slurm cluster

These scripts hold no site-specific values: those go into `lighthouse/local.env`, which git ignores.

## One-time setup

```bash
cp lighthouse/local.env.example lighthouse/local.env    # then fill it in
lighthouse/check_cluster.sh                              # read-only report; add --gpu for driver versions
```

## Download

Needs only `s5cmd` and `python3` on PATH; no conda environment.

```bash
lighthouse/download_data.sh --dry-run                    # show what would be fetched
lighthouse/download_data.sh                              # fetch; run it inside tmux, re-run to resume
```

| Part | Size | Goes to |
|---|---|---|
| training logs, without stereo cameras | about 146 GB for 150 logs | `$RV_RAW/train/` |
| validation logs: 20 for reporting (complete), 10 for tuning | about 33 GB | `$RV_RAW/val/` |
| pretrained weights | 3.5 GB | `$RV_SCRATCH/weights/` |

`RV_RAW` defaults to `$RV_SCRATCH/raw/argoverse2_sensor` and can be set in `local.env`.

Every log is checked against the bucket listing and every weight file against a pinned SHA-256. The check
tables are written next to the data as `*.download_check.tsv`; the script exits with an error while
anything is incomplete.

## Environment

One conda environment, `realityview`, with the same package versions as the development workstation.
Only the CUDA build of PyTorch depends on the cluster: pick it from the GPU driver that
`lighthouse/check_cluster.sh --gpu` reports.

```bash
lighthouse/setup_env.sh cu130            # driver 580 or newer; use cu126 for drivers 525 to 579
lighthouse/test_env.sh h100 --gsplat     # one short GPU allocation: torch sees the GPU, gsplat compiles and renders
```

gsplat compiles its CUDA code on first use with the compiler inside the environment. The check and the job
scripts set this up in one place (`rv_gsplat_env` in `common.sh`):

- The code is compiled for the L40S and the H100. For other GPUs set `RV_CUDA_ARCH_LIST` in `local.env`.
- The environment's own `libstdc++` is preloaded. Without it PyTorch loads the system's copy first, and on
  an older system (RHEL 8) the compiled extension fails to import with `CXXABI_1.3.15 not found`.

## Jobs

Everything after the download runs as a batch job. `submit.sh` snapshots the committed code, so a queued
job is not changed by a later `git pull`, and writes each job's output to `~/Reality_View/logs/`.

```bash
lighthouse/check_nas.sh                  # once: can results be copied to long-term storage (RV_NAS)?
lighthouse/submit.sh preprocess          # CPU: images, LiDAR depth, boxes, index
lighthouse/submit.sh sky --after <id>    # 1 GPU: sky scores, data checks, overlay pictures
lighthouse/submit.sh baseline            # 1 GPU: depth errors of the original network, several pose scales
lighthouse/submit.sh train --gpus 1 --hours 1 --name smoke -- total_steps=100 stop_at_step=50   # try it; twice
lighthouse/submit.sh train --hours 8     # 8 GPUs: train the backbone
lighthouse/setup_head.sh                 # once: lpips and its VGG weights for the Gaussian head
lighthouse/submit.sh head --hours 12     # 8 GPUs: train the Gaussian head on the trained backbone
lighthouse/submit.sh targets --gpus 1 --hours 1 --name targets_preview -- --preview   # candidate chase cameras
lighthouse/submit.sh targets -- --back 7 --up 4 --pitch 12 --hfov 70   # 8 GPUs: chase images of the training logs
```

A training job stops with a checkpoint 20 minutes before its time limit; submitting the same command
again continues it. `touch $RV_SCRATCH/runs/<name>/STOP` stops it earlier without losing work.
With `RV_NAS` set in `local.env`, checkpoints, numbers, logs, overlay and comparison pictures and the
rendered chase images are copied there through rclone; processed data stays on scratch (it can be made again).

## Scripts

| Script | What it does |
|---|---|
| `check_cluster.sh` | read-only report: nodes, GPU drivers, internet access, storage, existing data |
| `download_data.sh` | downloads raw data and weights for stage 1 |
| `setup_env.sh` | creates the conda environment `realityview` from `requirements.txt` |
| `test_env.sh` | checks the environment on a GPU node (runs `env_probe.py` through `srun`) |
| `submit_test_env.sh` | the same check as a queued batch job |
| `check_nas.sh` | checks that rclone can write to the place results are copied to |
| `setup_head.sh` | adds what the Gaussian head needs: `lpips` and the VGG weights it loads |
| `submit.sh` | submits a job: `preprocess`, `sky`, `baseline`, `train`, `head` or `targets` (bodies in `jobs/`) |
| `common.sh` | shared settings, sourced by the others |
