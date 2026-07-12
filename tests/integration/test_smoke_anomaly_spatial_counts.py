"""Smoke integration for anomaly, spatial, and count stages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crashlab.config import load_config
from crashlab.data.artifacts import processed_path
from crashlab.data.clean import split_clean_rejected
from crashlab.models.anomalies import run_anomaly_detection
from crashlab.models.counts import run_count_training
from crashlab.models.hotspots import run_hotspot_clustering
from crashlab.paths import ensure_dirs


def test_smoke_anomaly_spatial_counts_on_fixture(repo_root: Path | None = None) -> None:
    from crashlab.config import find_repo_root

    root = repo_root or find_repo_root()
    config = load_config("smoke", repo_root=root)
    paths = ensure_dirs(config)

    fixture = root / "data" / "samples" / "fixture.csv"
    raw = pd.read_csv(fixture, dtype=str, keep_default_na=False)
    clean, _ = split_clean_rejected(raw, year_start=2015, year_end=2023)
    clean.to_parquet(processed_path(paths, config.profile), index=False)

    anomaly = run_anomaly_detection(config, paths, force=True)
    spatial = run_hotspot_clustering(config, paths, force=True)
    counts = run_count_training(config, paths, force=True)

    assert anomaly["status"] == "completed"
    assert spatial["status"] == "completed"
    assert counts["status"] == "completed"
    assert "rules" in anomaly["methods_detail"]
    assert "all" in spatial["subsets"]
