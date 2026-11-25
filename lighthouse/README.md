# Running on a Slurm cluster

These scripts hold no site-specific values: those go into `lighthouse/local.env`, which git ignores.

## One-time setup

```bash
cp lighthouse/local.env.example lighthouse/local.env    # then fill it in
lighthouse/check_cluster.sh                              # read-only report; add --gpu for driver versions
lighthouse/setup_tools_env.sh                            # small conda environment with s5cmd
```

## Download

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

## Scripts

| Script | What it does |
|---|---|
| `check_cluster.sh` | read-only report: nodes, GPU drivers, internet access, storage, existing data |
| `setup_tools_env.sh` | creates the conda environment used for downloading |
| `download_data.sh` | downloads raw data and weights for stage 1 |
| `common.sh` | shared settings, sourced by the others |
