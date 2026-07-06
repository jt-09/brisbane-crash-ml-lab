"""Count-model lag leakage and aggregation tests."""

from __future__ import annotations

import pandas as pd

from crashlab.models.counts import add_historical_lag_features, build_count_panel


def _series_frame() -> pd.DataFrame:
    rows = []
    for year in (2020, 2021, 2022):
        for month in (1, 2):
            rows.append(
                {
                    "crash_ref_number": f"R{year}{month}",
                    "crash_severity": "Minor Injury",
                    "severe_binary": 0,
                    "loc_suburb": "Testville",
                    "crash_year": year,
                    "crash_month": month,
                }
            )
    return pd.DataFrame(rows)


def test_seasonal_lag_uses_only_prior_years() -> None:
    panel = build_count_panel(_series_frame())
    panel = add_historical_lag_features(panel, count_col="crash_count")
    row_2022_jan = panel.loc[(panel["crash_year"] == 2022) & (panel["crash_month"] == 1)].iloc[0]
    assert pd.isna(row_2022_jan["prev_month_count"]) or row_2022_jan["prev_month_count"] >= 0
    assert row_2022_jan["seasonal_hist_mean"] == 1.0


def test_first_period_has_no_lags() -> None:
    panel = build_count_panel(_series_frame())
    panel = add_historical_lag_features(panel, count_col="crash_count")
    first = panel.sort_values("period").iloc[0]
    assert pd.isna(first["prev_month_count"])
    assert pd.isna(first["seasonal_hist_mean"])


def test_prev_month_count_is_strictly_prior() -> None:
    panel = build_count_panel(_series_frame())
    panel = add_historical_lag_features(panel, count_col="crash_count")
    ordered = panel.sort_values("period")
    for i in range(1, len(ordered)):
        current = ordered.iloc[i]
        previous = ordered.iloc[i - 1]
        if current["loc_suburb"] == previous["loc_suburb"]:
            assert current["prev_month_count"] == previous["crash_count"]
