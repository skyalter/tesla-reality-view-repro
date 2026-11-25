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

## Scripts

| Script | What it does |
|---|---|
| `check_cluster.sh` | read-only report: nodes, GPU drivers, internet access, storage, existing data |
| `download_data.sh` | downloads raw data and weights for stage 1 |
| `setup_env.sh` | creates the conda environment `realityview` from `requirements.txt` |
| `test_env.sh` | checks the environment on a GPU node (runs `env_probe.py` through `srun`) |
| `common.sh` | shared settings, sourced by the others |
