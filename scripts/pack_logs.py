#!/usr/bin/env python3
"""Pack processed logs into one uncompressed zip each (for transfer), after their checks passed.

Usage (from the repository root):
    PYTHONPATH=. python scripts/pack_logs.py --out-root <processed>/av2_v1 --split val [--logs <list or ids>]

Writes <out-root>/logs/<split>/<log_id>.zip next to the loose directory, which is kept; the training reader
uses the directory when both exist. Logs without DONE or with failed checks are not packed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from realityview.data.splits import read_list  # noqa: E402
from realityview.data.store import pack_log  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default=os.environ.get("RV_PROCESSED_ROOT"), required=False)
    p.add_argument("--split", required=True)
    p.add_argument("--logs", default=None)
    args = p.parse_args()
    d = os.path.join(args.out_root, "logs", args.split)
    if args.logs is None:
        logs = sorted(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n)) and not n.endswith(".partial"))
    elif os.path.isfile(args.logs):
        logs = read_list(args.logs)
    else:
        logs = [x for x in args.logs.split(",") if x]
    bad = 0
    for log_id in logs:
        done = os.path.join(d, log_id, "DONE")
        if not os.path.isfile(done) or not json.load(open(done)).get("checks_passed"):
            print(f"{log_id}: not packed (no DONE or checks failed)")
            bad += 1
            continue
        n = pack_log(os.path.join(d, log_id), os.path.join(d, log_id + ".zip"))
        print(f"{log_id}: {n} files -> {log_id}.zip")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
