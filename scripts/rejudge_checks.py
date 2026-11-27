#!/usr/bin/env python
"""Apply the current check limits to logs that are already processed.

The per-log checks store the numbers they measured in CHECKS.json. When a limit in realityview/data/checks.py
changes, this script judges the stored numbers again and updates CHECKS.json and the ``checks_passed`` flag
(and the CHECKS.json hash) in DONE. Nothing else in a log is touched and nothing is processed again. The
rewritten CHECKS.json is byte for byte what processing the log with the current code would write.

    python scripts/rejudge_checks.py --out-root <processed root> [--split train] [--dry-run]

Exit status 1 when a log still fails afterwards.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from realityview.data.checks import rejudge_log_checks  # noqa: E402
from realityview.data.store import sha256_file  # noqa: E402


def _replace(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", required=True)
    p.add_argument("--split", default=None, help="only this split (default: every split)")
    p.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = p.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.out_root, "logs", args.split or "*", "*", "")))
    n = n_changed = n_failed = 0
    for d in dirs:
        checks_path, done_path = os.path.join(d, "CHECKS.json"), os.path.join(d, "DONE")
        if not (os.path.isfile(checks_path) and os.path.isfile(done_path)):
            continue
        n += 1
        tag = "/".join(d.rstrip(os.sep).split(os.sep)[-2:])
        with open(checks_path) as f:
            before = f.read()
        doc, changed = rejudge_log_checks(json.loads(before))
        after = json.dumps(doc, indent=1, sort_keys=True)
        with open(done_path) as f:
            done = json.load(f)
        failed = [k for k, v in doc["checks"].items() if v.get("passed") is False]
        n_failed += bool(failed)
        if failed:
            print(f"{tag}: still fails: {', '.join(failed)}", flush=True)
        if after == before and done.get("checks_passed") == doc["passed"]:
            continue
        n_changed += 1
        if changed:
            print(f"{tag}: {'passes' if doc['passed'] else 'fails'} now (verdict changed: {', '.join(changed)})",
                  flush=True)
        if args.dry_run:
            continue
        _replace(checks_path, after)
        done["checks_passed"] = bool(doc["passed"])
        if "CHECKS.json" in done.get("files", {}):
            done["files"]["CHECKS.json"] = sha256_file(checks_path)
        _replace(done_path, json.dumps(done, indent=1))
    print(f"rejudged {n} logs: {n_changed} {'would be ' if args.dry_run else ''}updated, {n_failed} still failing")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
