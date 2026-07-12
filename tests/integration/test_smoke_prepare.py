"""End-to-end smoke prepare pipeline (offline)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crashlab.config import load_config
from crashlab.data.acquire import run_acquire
from crashlab.data.artifacts import processed_path
from crashlab.data.clean import run_prepare
from crashlab.data.validate import run_validate
from crashlab.paths import ensure_dirs


@pytest.fixture
def repo_root() -> Path:
    from crashlab.config import find_repo_root

    return find_repo_root()


def test_smoke_fixture_to_parquet_offline(repo_root: Path, monkeypatch) -> None:
    monkeypatch.setenv("CRASHLAB_ALLOW_NETWORK", "0")
    config = load_config("smoke", repo_root=repo_root)
    paths = ensure_dirs(config)

    acquire_result = run_acquire(config, paths, force=True)
    validate_result = run_validate(config, paths, force=True)
    prepare_result = run_prepare(config, paths, force=True)

    assert acquire_result["status"] == "completed"
    assert acquire_result["raw_path"].endswith("fixture_smoke.csv")
    assert validate_result["status"] == "completed"
    assert prepare_result["status"] == "completed"
    assert prepare_result.get("features", {}).get("status") == "completed"
    assert prepare_result.get("eda", {}).get("status") == "completed"

    parquet = processed_path(paths, config.profile)
    assert parquet.is_file()
    df = pd.read_parquet(parquet)
    assert len(df) >= 15
    assert "crash_severity" in df.columns
    assert "Property Damage Only" not in set(df["crash_severity"].astype(str))
