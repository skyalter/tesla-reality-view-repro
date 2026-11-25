"""Shared helpers for the tests. Real-data tests read their paths from environment variables:

    AV2_RAW_ROOT       Argoverse 2 Sensor root (holding val/ and train/)
    RV_TEST_SPLIT      split of the real log to use (default: val)
    RV_TEST_LOG        log id of a complete log (default: the first log of data_lists/av2_val_report20.txt)
    RV_PROCESSED_ROOT  output root of scripts/preprocess.py that contains RV_TEST_LOG

Tests that need them are skipped when they are not set.
"""
from __future__ import annotations

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def real_log():
    """(raw log path, split, log id) or skip."""
    root = os.environ.get("AV2_RAW_ROOT")
    if not root:
        raise unittest.SkipTest("AV2_RAW_ROOT not set")
    split = os.environ.get("RV_TEST_SPLIT", "val")
    log_id = os.environ.get("RV_TEST_LOG")
    if not log_id:
        with open(os.path.join(REPO, "data_lists", "av2_val_report20.txt")) as f:
            log_id = f.readline().strip()
    path = os.path.join(root, split, log_id)
    if not os.path.isdir(path):
        raise unittest.SkipTest(f"{split}/{log_id} not found under AV2_RAW_ROOT")
    return path, split, log_id


def processed_root():
    root = os.environ.get("RV_PROCESSED_ROOT")
    if not root:
        raise unittest.SkipTest("RV_PROCESSED_ROOT not set")
    return root
