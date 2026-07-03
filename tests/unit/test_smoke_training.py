"""Smoke training on fixture features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crashlab.config import load_config
from crashlab.data.clean import split_clean_rejected
from crashlab.features.build import run_feature_build
from crashlab.models.binary import run_binary_training
from crashlab.models.multiclass import run_multiclass_training
from crashlab.paths import ensure_dirs


def test_smoke_binary_multiclass_on_fixture_features(repo_root: Path | None = None) -> None:
    from crashlab.config import find_repo_root

    root = repo_root or find_repo_root()
    config = load_config("smoke", repo_root=root)
    paths = ensure_dirs(config)

    fixture = root / "data" / "samples" / "fixture.csv"
    raw = pd.read_csv(fixture, dtype=str, keep_default_na=False)
    clean, _ = split_clean_rejected(raw, year_start=2015, year_end=2023)

    run_feature_build(config, paths, force=True, df=clean)
    binary_result = run_binary_training(config, paths, force=True)
    multi_result = run_multiclass_training(config, paths, force=True)

    assert binary_result["status"] == "completed"
    assert multi_result["status"] == "completed"
    assert binary_result["moments"]["context"]["champion"] is not None
