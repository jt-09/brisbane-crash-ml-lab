"""Unit tests for error analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crashlab.evaluation.error_analysis import (
    build_fp_fn_tables,
    subgroup_performance,
)


def test_fp_fn_tables_counts() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    proba = np.array([0.9, 0.1, 0.4, 0.8])
    meta = pd.DataFrame({"crash_year": [2020, 2021, 2022, 2023]})
    out = build_fp_fn_tables(y_true, y_pred, proba, meta, max_rows=10)
    assert out["n_false_positives"] == 1
    assert out["n_false_negatives"] == 1
    assert len(out["false_positives"]) == 1


def test_subgroup_small_n_caveat() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    proba = np.array([0.2, 0.8, 0.3, 0.4, 0.1, 0.9])
    groups = pd.Series(["a", "a", "b", "b", "b", "b"])
    rows = subgroup_performance(y_true, y_pred, proba, groups, min_n=5)
    small = [r for r in rows if r["subgroup"] == "a"][0]
    assert small["reliable"] is False
    assert small["caveat"] is not None
