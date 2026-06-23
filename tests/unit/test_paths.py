"""Path helper tests."""

from __future__ import annotations

from pathlib import Path

from crashlab.config import find_repo_root, load_config
from crashlab.paths import CrashlabPaths, ensure_dirs


def test_ensure_dirs_creates_expected_directories(tmp_path: Path) -> None:
    repo_root = find_repo_root()
    config = load_config("smoke", repo_root=tmp_path, configs_dir=repo_root / "configs")
    paths = ensure_dirs(config)

    for directory in paths.all_dirs():
        assert directory.is_dir()

    assert paths.manifests_dir == paths.artifacts_dir / "manifests"
    assert paths.models_dir == paths.artifacts_dir / "models"


def test_paths_from_real_repo() -> None:
    repo_root = find_repo_root()
    config = load_config("smoke", repo_root=repo_root)
    paths = CrashlabPaths.from_config(config)
    assert paths.repo_root == repo_root.resolve()
    assert paths.samples_dir.name == "samples"
