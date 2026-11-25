#!/usr/bin/env python3
"""Choose the validation logs used for tuning (the "dev" set), reproducibly.

The dev set is taken from the Argoverse 2 validation split, excluding the logs reserved for reporting.
Rule: within each city, order the remaining logs by sha256("av2-rv-dev:" + log_id); then take one log
per city in turn (cities in alphabetical order) until N logs are chosen.

Usage:
    python scripts/select_val_dev.py [--n 10] > data_lists/av2_val_dev10.txt
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

LISTS = Path(__file__).resolve().parent.parent / "data_lists"
SALT = "av2-rv-dev:"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--city-table", type=Path, default=LISTS / "av2_val_log_city.tsv")
    ap.add_argument("--exclude", type=Path, default=LISTS / "av2_val_report20.txt")
    args = ap.parse_args()

    reserved = {line.split()[0] for line in args.exclude.read_text().splitlines() if line.strip()}
    by_city: dict[str, list[str]] = defaultdict(list)
    for line in args.city_table.read_text().splitlines()[1:]:
        log_id, city = line.split("\t")[:2]
        if log_id not in reserved:
            by_city[city].append(log_id)
    for logs in by_city.values():
        logs.sort(key=lambda log_id: hashlib.sha256((SALT + log_id).encode()).hexdigest())

    chosen: list[str] = []
    rank = 0
    while len(chosen) < args.n and any(rank < len(v) for v in by_city.values()):
        for city in sorted(by_city):
            if rank < len(by_city[city]) and len(chosen) < args.n:
                chosen.append(by_city[city][rank])
        rank += 1
    print("\n".join(chosen))


if __name__ == "__main__":
    main()
