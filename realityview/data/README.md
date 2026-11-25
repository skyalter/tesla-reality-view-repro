# Stage 1 data processing (Argoverse 2 Sensor)

Offline processing of the raw logs into small images, per-image LiDAR depth, interpolated boxes and a
moment index, plus the first version of the stage-1 training reader. Nothing is installed: run from the
repository root with `PYTHONPATH=.`. Paths always come from arguments or environment variables
(`AV2_RAW_ROOT`, `RV_PROCESSED_ROOT`).

## Commands

```bash
export PYTHONPATH=.
# process logs (finished logs are skipped on a rerun; incomplete raw logs are reported and skipped)
python scripts/preprocess.py --raw-root "$AV2_RAW_ROOT" --out-root "$RV_PROCESSED_ROOT" \
    --split val --logs data_lists/av2_val_report20.txt --workers 8 [--parallel auto|logs|images] [--first N] [--redo]
# collect per-log checks, check the split lists, draw overlays, read samples, reprocess one log and compare
python scripts/check_data.py --out-root "$RV_PROCESSED_ROOT" --split val --overlays [--overlay-dir DIR] \
    --read-samples 500 [--require-sky] [--repro <log_id> --raw-root "$AV2_RAW_ROOT"]
# sky scores for every stored image (needs a GPU and the DA3METRIC-LARGE weights); then run check_data.py again
python scripts/sky_scores.py --out-root "$RV_PROCESSED_ROOT" --split val --weights <weights>/da3metric-large/model.safetensors
# optional: one uncompressed zip per checked log, for transfer (after the sky scores)
python scripts/pack_logs.py --out-root "$RV_PROCESSED_ROOT" --split val
# unit tests (standard library unittest; real-data tests run only when the variables are set)
AV2_RAW_ROOT=... RV_PROCESSED_ROOT=... python -m unittest discover -s tests
```

## What is computed

| Step | Module | Rule |
|---|---|---|
| Moments | `index.py` | A moment is a front-center frame i; each other camera takes its nearest frame (they expose up to ~22.5 ms apart). t − 0.5 s = frame i − 10. `valid_stage1`: i ≥ 10 and the moments i and i − 10 complete (all cameras within 25 ms; the design said 30 ms, which lets the first or last moment of a log pair a camera with a frame 27.5 ms away from the neighbouring period). `valid_stage2`: also 20 ≤ i ≤ n − 21. Also stored: ego displacement t vs t − 0.5 s, max displacement within ±1 s. |
| Poses | `av2_io.py` | Exact lookup by timestamp for every image, sweep and box, float64. The reference frame is chosen at read time. |
| Images | `images.py` | Full decode, `INTER_AREA`, JPEG q95. Landscape cameras 2048×1550 → 518×392; front-center 1550×2048 → 518×684 whole, cut at read time. |
| Intrinsics | `images.py` | Integer pixel-centre convention: `fx' = fx·sx`, `cx' = (cx + 0.5)·sx − 0.5` (same for y). Front-center crop: rows 146..538, `cy −= 146`. A renderer with centres at j + 0.5 (gsplat) needs `intrinsics_for_half_pixel_renderer` (cx, cy + 0.5). |
| LiDAR depth | `lidar_depth.py` | Points are already compensated to the sweep start; per sweep: point → city (sweep-time pose) → ego at camera time → camera → K. `offset_ns` is never used to move points. Accumulated layer: static points of sweeps k0 − 5..k0 + 5 plus, inside dynamic boxes (grown by 0.3 m per side in length and width, 0.2 m per side in height), only points measured within 25 ms of the camera (sweeps k0 − 1..k0 + 1). Single layer: static points of k0 plus the same synchronised points. Nearest point per pixel, 0.5–250 m, ego-body pixels dropped, flags 1 = sweep k0, 2 = synchronised dynamic point, 4 = suspected see-through (> 1.2 × the 5×5 minimum). |
| Boxes | `boxes.py` | Boxes to the city frame with their own sweep pose; speed from centres ~1 s apart; moving if > 0.5 m/s or unknown; person-like categories always dynamic. Interpolated to every camera image (centre linear, orientation slerp, ≤ 50 ms extrapolation) and stored; masks are drawn at read time (`draw_box_mask`: box grown by the same per-side margins and swept by velocity × 0.1 s, clipped at a near plane, its exact outline projected and filled, dilated 2 px; boxes partly outside the image or behind the camera are drawn). |
| Checks | `checks.py` | Per log, into `CHECKS.json`: completeness (and byte sizes against `MANIFEST.tsv` if present), timestamps (every moment but the first and last complete), image cache, intrinsics, two-LiDAR agreement, coverage, motion, ego-mask points, depth files. `check_data.py` adds split-list overlap (data_lists/ and the root's splits/), `splits/val_fixed.tsv`, overlays, reading, reproducibility. |
| Reader | `dataset.py` | `Stage1Dataset`: 14 images (7 cameras × {t, t − 10}), K, `ref_T_cam` (reference = front-center camera at t, always), `dt`, dense depth + flags, ego masks, optional single layer, box masks and sky scores. Switches: `input_offsets`, `front_center_mode` ("crop" rows 146..538; "split" two views rows 0..392 and 292..684; "letterbox" 297×392 padded to 518, padding marked in `ego_mask`). Asserts the root's train/dev/val lists are disjoint. |

**Sky scores** are a second pass over the stored images (`scripts/sky_scores.py`, one GPU): the raw sky
output of DA3METRIC-LARGE, times 100, as uint8 PNG at the stored size (0 on ground and buildings, around 100
on sky; not a probability). The portrait front-center image is padded from 684 to 686 rows for the network
and cut back. No threshold is stored: the loss treats a score below `sky.low` (default 0.3) as not sky,
from `sky.high` (0.5) on as sky, and leaves the band in between alone. `check_data.py` checks that every
image has a score of its own size and that the bottom eighth of the pictures holds no sky, and draws a sky
overlay; the reader returns the scores with `with_sky=True`.

## Output layout

```
<out-root>/                       one root per processing config (e.g. av2_v1)
  CONFIG.yaml  CONFIG.sha256      all parameters; a root never mixes configs
  CODE_COMMIT.txt                 commits that wrote into this root
  ego_masks/<cam>.png             ego-body masks at the stored resolution (dilated by 1 px), all 7 cameras
  splits/                         train.txt dev.txt val.txt overfit.txt train_order.tsv (copied from
                                  data_lists/ after a disjointness check), val_fixed.tsv (check_data.py)
  runs/preprocess_<split>_<utc>.json   per-run report (status, timings, sizes of every log)
  checks_<split>.json             written by check_data.py
  overlays/<split>/<log>_i<moment>_{lidar,lidar_single,dyn,sky}.jpg   (or --overlay-dir)
  logs/<split>/<log_id>/          (or <log_id>.zip, same content)
    meta.json        city, sizes, K_full, K_stored, ego_T_cam, ego_T_lidar, config + sha256, code commit
    frames.parquet   per image: cam, frame, timestamp_ns, exact ego pose (qw..qz, tx..tz), k0, point counts
    moments.parquet  per front-center frame: idx_/ts_/dt_ms_/k0_<cam>, complete, valid_stage1/2, displacements
    sweeps.parquet   per sweep: timestamp, ego pose, number of boxes
    boxes.parquet    per image: interpolated boxes (city frame), velocity, moving, person_like, dynamic, extrapolated
    img/<cam>/<timestamp_ns>.jpg
    depth/acc/<cam>/<timestamp_ns>.npz      keys v, u (uint16), z (float16, m), flag (uint8)
    depth/single/<cam>/<timestamp_ns>.npz
    sky/<cam>/<timestamp_ns>.png            raw sky score x 100, uint8 (scripts/sky_scores.py); sky/DONE when complete
    CHECKS.json      per-log check results
    DONE             sha256 of every file, timings, sizes; written last (a log without DONE is unfinished)
```

A log is written to `<log_id>.partial/` and renamed when complete, so an interrupted run leaves no
half-finished log behind; rerunning resumes with the logs that have no DONE. Worker processes are always
spawned, never forked. With `--workers N` and at least N logs to do, whole logs run in parallel (one log
per process); otherwise the images of one log are shared out.

Pixel convention: stored intrinsics put pixel centres at integer coordinates. A renderer with centres at
j + 0.5 (gsplat) gets `images.intrinsics_for_half_pixel_renderer(K)` (cx, cy + 0.5); the LiDAR projection
and the reader keep the integer convention.

## Reader for the Gaussian head

`head_dataset.py`: `Stage1HeadDataset` (a `Stage1Dataset`) adds one target moment t + k per sample, whose
pictures the rendered Gaussians are compared with. Moments: `valid_stage2`, valid for `input_offsets`, and
`disp_win_max_m >= min_displacement_m` (default 0.5 m; 0 keeps standing moments). Targets, in front-center
frames:

- `target_offsets="random"` (training): one sample per moment; k among the integers in
  [−`max_target_frames`, `max_target_frames`] (default 20) that differ from every input offset by more than
  `min_gap_frames` (default 3) and whose moment is complete (27 candidates with the defaults). The pick is
  sha256(`"{seed}:{log_id}:{i}"`) modulo the number of candidates, so an index always gives the same sample.
- `target_offsets=[k1, k2, ...]` (evaluation): one sample per (moment, k) with t + k complete; k = 0 or an
  input offset repeats that input moment.

`samples[idx]` stays (log index, moment); `sample_offsets[idx]` is k. A sample holds every key of
`Stage1Dataset` (`box_mask` always, `sky` with `with_sky`, default on), plus:

| Key | Type and shape | Meaning |
|---|---|---|
| `ref_T_ego` | float32 (4,4) | ego frame (rear axle, x forward, y left, z up) at the front-center timestamp of t → reference; the inverse of the front-center extrinsics |
| `tgt_images`, `tgt_K`, `tgt_ref_T_cam`, `tgt_dt`, `tgt_cam_id`, `tgt_view`, `tgt_timestamp_ns` | as the inputs, first dimension M (7; 8 for "split") | the M views of t + k in `CAMS` order; same reference (front-center camera at t), `tgt_dt` relative to t |
| `tgt_depth`, `tgt_depth_flag` | float32 / uint8 (M,392,518) | LiDAR depth of `depth_layer` |
| `tgt_ego_mask` | bool (M,392,518) | ego body |
| `tgt_box_mask` | uint8 (M,392,518) | boxes of the target moment: 1 moving, 2 person-like, 4 any box |
| `tgt_sky` | float32 (M,392,518) | raw sky score (`with_sky`) |
| `tgt_offset` | int64 () | k |

Every target key equals what `Stage1Dataset` returns for the same moment read as an input (same
`front_center_mode`, same reference).
