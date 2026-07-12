"""Leakage denylist enforcement tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crashlab.config import load_config
from crashlab.data.clean import split_clean_rejected
from crashlab.features.build import (
    assert_valid_feature_columns,
    build_moment_features,
    run_feature_build,
)
from crashlab.features.constants import LEAKAGE_DENYLIST
from crashlab.features.temporal import compute_year_splits
from crashlab.paths import ensure_dirs


def _modeling_frame() -> pd.DataFrame:
    from crashlab.config import find_repo_root

    fixture = find_repo_root() / "data" / "samples" / "fixture.csv"
    raw = pd.read_csv(fixture, dtype=str, keep_default_na=False)
    clean, _ = split_clean_rejected(raw, year_start=2015, year_end=2023)
    return clean


def test_leakage_denylist_blocks_valid_moments() -> None:
    for moment in ("context", "triage"):
        with pytest.raises(ValueError, match="Leakage denylist"):
            assert_valid_feature_columns(["crash_hour_sin", "count_casualty_total"], moment)  # type: ignore[arg-type]


def test_context_excludes_triage_only_columns() -> None:
    with pytest.raises(ValueError, match="excluded"):
        assert_valid_feature_columns(["crash_nature__Rear-end"], "context")


def test_valid_context_features_pass() -> None:
    assert_valid_feature_columns(
        ["crash_hour_sin", "loc_suburb__Brisbane City", "is_weekend"],
        "context",
    )


def test_leakage_demo_allows_casualty_columns() -> None:
    names = ["count_casualty_total", "count_casualty_fatality"]
    assert_valid_feature_columns(names, "leakage_demo")


def test_feature_pipeline_excludes_denylist() -> None:
    df = _modeling_frame()
    years = sorted(int(y) for y in df["crash_year"].dropna().unique())
    splits = compute_year_splits(years, train_year_end=2021, val_years=[2022], test_years=[2023])

    for moment in ("context", "triage"):
        matrices, _, feature_names = build_moment_features(df, moment, splits, min_count=1)  # type: ignore[arg-type]
        assert matrices, f"expected matrices for {moment}"
        lowered = {n.lower() for n in feature_names}
        for denied in LEAKAGE_DENYLIST:
            assert denied not in lowered
        for name in feature_names:
            assert "casualty" not in name.lower()


def test_leakage_demo_includes_casualty_features() -> None:
    df = _modeling_frame()
    years = sorted(int(y) for y in df["crash_year"].dropna().unique())
    splits = compute_year_splits(years, train_year_end=2021, val_years=[2022], test_years=[2023])
    _, _, names = build_moment_features(df, "leakage_demo", splits, min_count=1)
    assert any("count_casualty" in n for n in names)


def test_feature_matrices_contain_no_nan() -> None:
    df = _modeling_frame()
    years = sorted(int(y) for y in df["crash_year"].dropna().unique())
    splits = compute_year_splits(years, train_year_end=2021, val_years=[2022], test_years=[2023])

    for moment in ("context", "triage"):
        matrices, bundle, _ = build_moment_features(df, moment, splits, min_count=1)  # type: ignore[arg-type]
        assert bundle.imputer_ is not None
        for split_name, matrix in matrices.items():
            assert not matrix.isna().any().any(), f"NaN in {moment}/{split_name}"
            values = matrix.to_numpy(dtype=float)
            assert np.isfinite(values).all(), f"non-finite values in {moment}/{split_name}"


def test_run_feature_build_offline(repo_root: Path | None = None) -> None:
    from crashlab.config import find_repo_root

    root = repo_root or find_repo_root()
    config = load_config("smoke", repo_root=root)
    paths = ensure_dirs(config)
    df = _modeling_frame()
    result = run_feature_build(config, paths, force=True, df=df)
    assert result["status"] == "completed"
    assert "context" in result["moments"]
