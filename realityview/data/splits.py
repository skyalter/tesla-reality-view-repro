"""Log lists and the rule that the training, tuning and reporting lists never overlap."""
from __future__ import annotations

import itertools
import os


def read_list(path: str) -> list[str]:
    """One log id per line; also accepts a TSV with a ``log_id`` column. Blank lines and # comments skipped."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if lines and "\t" in lines[0]:
        head = lines[0].split("\t")
        col = head.index("log_id")
        return [ln.split("\t")[col] for ln in lines[1:]]
    return lines


def overlaps(lists: dict[str, list[str]]) -> dict[str, list[str]]:
    """Pairwise intersections; empty dict means all lists are disjoint."""
    out = {}
    for (a, la), (b, lb) in itertools.combinations(lists.items(), 2):
        inter = sorted(set(la) & set(lb))
        if inter:
            out[f"{a} & {b}"] = inter
    return out


def assert_disjoint(lists: dict[str, list[str]]):
    bad = overlaps(lists)
    if bad:
        raise ValueError(f"log lists overlap: { {k: len(v) for k, v in bad.items()} }")


def default_lists(repo_root: str) -> dict[str, list[str]]:
    d = os.path.join(repo_root, "data_lists")
    return {
        "train_stage1": read_list(os.path.join(d, "av2_train_stage1.txt")),
        "val_dev10": read_list(os.path.join(d, "av2_val_dev10.txt")),
        "val_report20": read_list(os.path.join(d, "av2_val_report20.txt")),
    }
