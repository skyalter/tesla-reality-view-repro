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


SPLIT_FILES = {  # file in <processed root>/splits/ -> role
    "train.txt": "training (stage 1)",
    "dev.txt": "tuning",
    "val.txt": "reporting",
    "overfit.txt": "overfitting check (first logs of the training order; part of train)",
}


def overfit_logs(repo_root: str, n: int = 2) -> list[str]:
    return read_list(os.path.join(repo_root, "data_lists", "av2_train_order.tsv"))[:n]


def write_split_files(processed_root: str, repo_root: str) -> dict[str, list[str]]:
    """Copy the log lists into ``<processed_root>/splits/`` (train, dev, val, overfit, train_order) after
    checking that training, tuning and reporting lists are pairwise disjoint. Existing files must match."""
    import shutil

    lists = default_lists(repo_root)
    assert_disjoint(lists)
    out = {"train.txt": lists["train_stage1"], "dev.txt": lists["val_dev10"], "val.txt": lists["val_report20"],
           "overfit.txt": overfit_logs(repo_root)}
    if not set(out["overfit.txt"]) <= set(out["train.txt"]):
        raise ValueError("overfit logs must be training logs")
    d = os.path.join(processed_root, "splits")
    os.makedirs(d, exist_ok=True)
    for name, ids in out.items():
        text = "".join(f"{x}\n" for x in ids)
        f = os.path.join(d, name)
        if os.path.isfile(f):
            with open(f) as fh:
                if fh.read() != text:
                    raise ValueError(f"{f} differs from data_lists/; a processed root keeps its lists")
        else:
            with open(f, "w") as fh:
                fh.write(text)
    src = os.path.join(repo_root, "data_lists", "av2_train_order.tsv")
    dst = os.path.join(d, "train_order.tsv")
    if not os.path.isfile(dst):
        shutil.copyfile(src, dst)
    return out


def check_root_splits(processed_root: str) -> dict[str, list[str]]:
    """Read ``<processed_root>/splits/`` and assert train, dev and val are pairwise disjoint (to be called when
    training starts). Returns the lists; an empty dict if the root has no splits/ directory."""
    d = os.path.join(processed_root, "splits")
    if not os.path.isdir(d):
        return {}
    lists = {n[:-4]: read_list(os.path.join(d, n)) for n in ("train.txt", "dev.txt", "val.txt")
             if os.path.isfile(os.path.join(d, n))}
    assert_disjoint(lists)
    return lists


def val_fixed_moments(n_frames: int, n: int = 8, margin: int = 20) -> list[int]:
    """Fixed evaluation moments of a log with ``n_frames`` front-center frames:
    t_k = 20 + round(k * (n_frames - 41) / 7), k = 0..7 (round half up)."""
    import math

    return [margin + int(math.floor(k * (n_frames - 2 * margin - 1) / (n - 1) + 0.5)) for k in range(n)]
