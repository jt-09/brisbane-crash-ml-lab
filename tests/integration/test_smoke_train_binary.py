"""Integration: smoke prepare then train-binary."""

from __future__ import annotations

from pathlib import Path

import pytest

from crashlab.config import load_config
from crashlab.data.clean import run_prepare
from crashlab.models.binary import run_binary_training
from crashlab.paths import ensure_dirs


@pytest.fixture
def repo_root() -> Path:
    from crashlab.config import find_repo_root

    return find_repo_root()


def test_smoke_train_binary_after_prepare(repo_root: Path) -> None:
    config = load_config("smoke", repo_root=repo_root)
    paths = ensure_dirs(config)
    prepare_result = run_prepare(config, paths, force=True)
    assert prepare_result["status"] == "completed"

    train_result = run_binary_training(config, paths, force=True)
    assert train_result["status"] == "completed"
    assert "context" in train_result["moments"]
    context = train_result["moments"]["context"]
    assert context.get("champion") is not None
    assert context["champion"]["model_name"] in {"dummy", "logistic"}

    leakage = train_result["moments"].get("leakage_demo", {})
    assert leakage.get("excluded_from_leaderboard") is True
    assert leakage.get("champion") is None
