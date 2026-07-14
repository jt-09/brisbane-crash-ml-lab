"""Anomaly detection determinism and stability tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crashlab.models.anomalies import (
    isolation_forest_scores,
    robust_z_anomaly_scores,
    rule_anomaly_scores,
)


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crash_ref_number": [f"T{i:03d}" for i in range(8)],
            "crash_year": [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023],
            "crash_month": [1, 2, 1, 2, 1, 2, 1, 2],
            "crash_hour": [8, 9, 10, 11, 12, 13, 22, 23],
            "crash_longitude": np.linspace(153.0, 153.07, 8),
            "crash_latitude": np.linspace(-27.45, -27.52, 8),
            "crash_speed_limit": [60, 60, 50, 50, 70, 70, 100, 40],
            "loc_suburb": ["A", "A", "B", "B", "A", "B", "C", "C"],
            "crash_severity": ["Minor Injury"] * 8,
        }
    )


def test_rule_scores_are_deterministic() -> None:
    df = _tiny_frame()
    scores_a, _ = rule_anomaly_scores(df)
    scores_b, _ = rule_anomaly_scores(df)
    np.testing.assert_array_equal(scores_a, scores_b)


def test_isolation_forest_deterministic_for_fixed_seed() -> None:
    df = _tiny_frame()
    scores_a = isolation_forest_scores(df, seed=42, n_estimators=50)
    scores_b = isolation_forest_scores(df, seed=42, n_estimators=50)
    np.testing.assert_allclose(scores_a, scores_b)


def test_robust_z_flags_extreme_speed() -> None:
    df = _tiny_frame()
    scores, reasons = robust_z_anomaly_scores(df)
    speed_idx = int(df.index[df["crash_speed_limit"] == 100][0])
    assert scores[speed_idx] >= 3.0
    flat_reasons = [item for group in reasons for item in group]
    assert any("crash_speed_limit" in r for r in flat_reasons)
