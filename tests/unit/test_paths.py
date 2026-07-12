"""Path helper tests."""

from __future__ import annotations

from pathlib import Path

from crashlab.config import find_repo_root, load_config
from crashlab.data.artifacts import processed_path
from crashlab.features.build import encoder_path, features_dir, matrix_path
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


def test_profile_artifact_paths_are_isolated() -> None:
    repo_root = find_repo_root()
    smoke = load_config("smoke", repo_root=repo_root)
    standard = load_config("standard", repo_root=repo_root)
    paths = CrashlabPaths.from_config(smoke)

    smoke_processed = processed_path(paths, smoke.profile)
    standard_processed = processed_path(paths, standard.profile)
    assert smoke_processed != standard_processed
    assert smoke.profile in str(smoke_processed)
    assert standard.profile in str(standard_processed)

    smoke_matrix = matrix_path(paths, smoke.profile, "context", "train")
    standard_matrix = matrix_path(paths, standard.profile, "context", "train")
    assert smoke_matrix != standard_matrix
    assert smoke_matrix.parent == features_dir(paths, smoke.profile)
    assert standard_matrix.parent == features_dir(paths, standard.profile)

    smoke_encoder = encoder_path(paths, smoke.profile, "context")
    standard_encoder = encoder_path(paths, standard.profile, "context")
    assert smoke_encoder != standard_encoder

    smoke_model = paths.models_dir / smoke.profile / "binary_context_logistic.joblib"
    standard_model = paths.models_dir / standard.profile / "binary_context_logistic.joblib"
    assert smoke_model != standard_model
    assert smoke_model.parent.name == smoke.profile
    assert standard_model.parent.name == standard.profile
