#!/usr/bin/env python3
"""Rank all Argoverse 2 training logs once, in a fixed order; the first N logs are the N-log training set.

Because the order is fixed, a larger training set always contains the smaller ones.

Rule
1. Drop logs in which the car drives less than MIN_PATH_M metres (it is practically parked).
2. Give every log a fixed pseudo-random key: sha256("<seed>:<log_id>").
3. Inside a city, split the logs into slow / medium / fast by mean speed and order each group by key.
4. Hand out places one at a time with the Sainte-Lague method, at two levels with the same rule: across
   cities, and inside a city across the three speed groups. Each city or group is weighted by
   (number of eligible logs) ** SHARE_POWER. A power below 1 gives small cities and rare speeds a somewhat
   larger share than their size, without letting them dominate.

Usage:
    python scripts/select_train_logs.py --data-root <av2 sensor root> [--n 150]

Needs only the pose files: <data-root>/train/<log_id>/city_SE3_egovehicle.feather
(download them with: scripts/av2_download.sh --split train --list data_lists/av2_train_log_city.tsv
 --dest <data-root> --meta-only).

Writes data_lists/av2_train_order.tsv (all logs, ranked, with the numbers used) and
data_lists/av2_train_stage1.txt (the first N log ids).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LISTS = Path(__file__).resolve().parent.parent / "data_lists"
SEED = "av2-rv-2026"
MIN_PATH_M = 10.0
SHARE_POWER = 0.75
SPEED_EDGES_MPS = (3.0, 7.0)
STRATA = ("slow", "medium", "fast")


def pose_metrics(pose_file: Path) -> dict[str, float]:
    """Driven distance, mean speed and how often the car is standing still, from the ego poses."""
    df = pd.read_feather(pose_file, columns=["timestamp_ns", "tx_m", "ty_m"])
    df = df.sort_values("timestamp_ns").drop_duplicates("timestamp_ns")
    t = (df.timestamp_ns.to_numpy() - df.timestamp_ns.iloc[0]) * 1e-9
    # Resample to 10 Hz so that the pose rate (which follows the sensors) does not matter.
    grid = np.arange(0.0, t[-1], 0.1)
    x = np.interp(grid, t, df.tx_m.to_numpy())
    y = np.interp(grid, t, df.ty_m.to_numpy())
    path = float(np.hypot(np.diff(x), np.diff(y)).sum())
    starts = np.arange(0, len(grid) - 10, 5)  # 1 s windows, every 0.5 s
    moved = np.hypot(x[starts + 10] - x[starts], y[starts + 10] - y[starts])
    return {
        "duration_s": float(t[-1]),
        "path_m": path,
        "mean_speed_mps": path / float(t[-1]),
        "still_fraction": float((moved < 0.5).mean()),
    }


def stratum(speed: float) -> str:
    return STRATA[int(speed >= SPEED_EDGES_MPS[0]) + int(speed >= SPEED_EDGES_MPS[1])]


def interleave(pools: dict[str, list[str]], order: list[str]) -> list[str]:
    """Merge named lists into one sequence, one item at a time, by the Sainte-Lague method.

    Every prefix of the result holds the pools roughly in proportion to len(pool) ** SHARE_POWER.
    Ties go to the name that comes last in `order`.
    """
    weight = {name: len(items) ** SHARE_POWER for name, items in pools.items()}
    taken = dict.fromkeys(pools, 0)
    merged: list[str] = []
    for _ in range(sum(len(items) for items in pools.values())):
        open_names = [name for name in pools if taken[name] < len(pools[name])]
        name = max(open_names, key=lambda n: (weight[n] / (2 * taken[n] + 1), order.index(n)))
        merged.append(pools[name][taken[name]])
        taken[name] += 1
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--city-table", type=Path, default=LISTS / "av2_train_log_city.tsv")
    ap.add_argument("--out-dir", type=Path, default=LISTS)
    args = ap.parse_args()

    table = pd.read_csv(args.city_table, sep="\t")[["log_id", "city"]]
    rows, missing = [], []
    for log_id in table.log_id:
        pose_file = args.data_root / "train" / log_id / "city_SE3_egovehicle.feather"
        if pose_file.is_file():
            rows.append({"log_id": log_id, **pose_metrics(pose_file)})
        else:
            missing.append(log_id)
    if missing:
        print(f"{len(missing)} logs have no pose file under {args.data_root}/train, e.g. {missing[0]}", file=sys.stderr)
        print("the ranking must see all logs; download the metadata first", file=sys.stderr)
        return 1
    table = table.merge(pd.DataFrame(rows), on="log_id")
    table["eligible"] = table.path_m >= MIN_PATH_M
    table["speed"] = table.mean_speed_mps.map(stratum)
    table["key"] = [hashlib.sha256(f"{SEED}:{log_id}".encode()).hexdigest()[:16] for log_id in table.log_id]

    # Inside each city: the three speed groups, each ordered by key, merged into one sequence.
    city_order: dict[str, list[str]] = {}
    for city, group in table[table.eligible].groupby("city"):
        pools = {s: list(g.sort_values("key").log_id) for s, g in group.groupby("speed")}
        city_order[city] = interleave(pools, list(STRATA))

    # Across cities: the same merge.
    ranked = interleave(city_order, sorted(city_order))

    table["rank"] = table.log_id.map({log_id: i + 1 for i, log_id in enumerate(ranked)})
    table = table.sort_values(["rank", "log_id"], na_position="last")
    columns = ["rank", "log_id", "city", "speed", "path_m", "mean_speed_mps", "still_fraction", "duration_s", "eligible"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = table[columns].copy()
    out["rank"] = out["rank"].astype("Int64")
    out.to_csv(args.out_dir / "av2_train_order.tsv", sep="\t", index=False, float_format="%.2f")
    chosen = table[table["rank"] <= args.n]
    (args.out_dir / "av2_train_stage1.txt").write_text("\n".join(chosen.log_id) + "\n")

    print(f"eligible {int(table.eligible.sum())} of {len(table)}; dropped {int((~table.eligible).sum())} that drive under {MIN_PATH_M:.0f} m")
    print(f"first {len(chosen)} logs by city:  " + ", ".join(f"{c} {n}" for c, n in chosen.groupby("city").size().items()))
    print(f"first {len(chosen)} logs by speed: " + ", ".join(f"{s} {int((chosen.speed == s).sum())}" for s in STRATA))
    print(f"driving time {chosen.duration_s.sum() / 60:.1f} min, distance {chosen.path_m.sum() / 1000:.1f} km")
    return 0


if __name__ == "__main__":
    sys.exit(main())
