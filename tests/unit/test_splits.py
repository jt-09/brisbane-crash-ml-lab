"""Temporal split and encoder fit-scope tests."""

from __future__ import annotations

import pandas as pd

from crashlab.features.encoders import fit_encoder_bundle, transform_with_mixed_encoding
from crashlab.features.temporal import YearSplits, assign_split_column, compute_year_splits


def test_year_splits_no_overlap() -> None:
    splits = compute_year_splits(
        [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
        train_year_end=2021,
        val_years=[2022],
        test_years=[2023],
    )
    splits.validate_disjoint()
    assert splits.train_years == frozenset(range(2015, 2022))
    assert splits.val_years == frozenset({2022})
    assert splits.test_years == frozenset({2023})


def test_proportional_splits_for_short_span() -> None:
    splits = compute_year_splits([2019, 2020, 2021, 2022, 2023])
    splits.validate_disjoint()
    assert len(splits.train_years | splits.val_years | splits.test_years) == 5


def test_assign_split_column_maps_whole_years() -> None:
    splits = YearSplits(
        train_years=frozenset({2020, 2021}),
        val_years=frozenset({2022}),
        test_years=frozenset({2023}),
    )
    df = pd.DataFrame({"crash_year": [2020, 2022, 2023, 2019]})
    assigned = assign_split_column(df, splits)
    assert list(assigned) == ["train", "val", "test", None]


def test_encoders_fit_train_only() -> None:
    train = pd.DataFrame(
        {
            "loc_suburb": ["A", "A", "B", "B", "C"],
            "crash_hour_sin": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    test = pd.DataFrame(
        {
            "loc_suburb": ["D", "A"],
            "crash_hour_sin": [0.9, 0.1],
        }
    )
    bundle = fit_encoder_bundle(
        train,
        ["loc_suburb"],
        ["crash_hour_sin"],
        min_count=2,
    )
    transform_with_mixed_encoding(bundle, train)
    test_x = transform_with_mixed_encoding(bundle, test)
    suburb_cols = [c for c in test_x.columns if c.startswith("loc_suburb")]
    assert suburb_cols
    assert test_x.loc[test.index[0], suburb_cols[0]] >= 0.0
