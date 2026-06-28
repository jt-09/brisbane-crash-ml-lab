"""Configuration loading and inheritance tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from crashlab.config import find_repo_root, load_config


@pytest.fixture
def repo_root() -> Path:
    return find_repo_root()


def test_load_smoke_profile(repo_root: Path) -> None:
    config = load_config("smoke", repo_root=repo_root)
    assert config.profile == "smoke"
    assert config.use_fixture is True
    assert config.fixture_path is not None
    assert Path(config.fixture_path) == (repo_root / "data" / "samples" / "fixture.csv").resolve()
    assert config.fixture_raw_path is not None
    assert (
        Path(config.fixture_raw_path)
        == (repo_root / "data" / "raw" / "fixture_smoke.csv").resolve()
    )
    assert config.project["seed"] == 42
    assert config.data["lga"] == "Brisbane City"
    assert config.models["binary"] == ["dummy", "logistic"]
    assert config.tuning["n_iter"] == 0


def test_load_standard_profile(repo_root: Path) -> None:
    config = load_config("standard", repo_root=repo_root)
    assert config.profile == "standard"
    assert config.use_fixture is False
    assert "hist_gradient_boosting" in config.models["binary"]
    assert config.tuning["n_iter"] == 15
    assert config.tuning["n_folds"] == 3


def test_load_extended_inherits_standard(repo_root: Path) -> None:
    config = load_config("extended", repo_root=repo_root)
    assert config.profile == "extended"
    assert config.tuning["n_iter"] == 30
    assert config.tuning["n_folds"] == 5
    assert config.tuning.get("extra_seeds") is True
    assert "hist_gradient_boosting" in config.models["binary"]


def test_paths_resolved_relative_to_repo(repo_root: Path) -> None:
    config = load_config("smoke", repo_root=repo_root)
    raw_dir = Path(config.paths["raw_dir"])
    assert raw_dir.is_absolute()
    assert raw_dir == (repo_root / "data" / "raw").resolve()


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        load_config("nonexistent")
