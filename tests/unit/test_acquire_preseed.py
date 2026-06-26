"""Preseed hash reuse and offline acquisition behaviour."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from crashlab.config import load_config
from crashlab.data.acquire import (
    AcquisitionError,
    default_raw_path,
    fixture_acquire_path,
    run_acquire,
    sha256_file,
)
from crashlab.data.schema import PRESEED_SHA256
from crashlab.paths import ensure_dirs


@pytest.fixture
def repo_root() -> Path:
    from crashlab.config import find_repo_root

    return find_repo_root()


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    payload = b"brisbane-preseed-test\n"
    path = tmp_path / "sample.csv"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest().upper()
    assert sha256_file(path) == expected


def test_preseed_short_circuit_reuses_matching_file(tmp_path: Path, repo_root: Path) -> None:
    preseed = tmp_path / "brisbane_crashes_2015_2023.csv"
    preseed.write_text("crash_ref_number,crash_severity\n1,Minor Injury\n", encoding="utf-8")
    digest = sha256_file(preseed)

    config = load_config("standard", repo_root=repo_root)
    config.raw["preseed_raw"] = {
        "path": str(preseed),
        "sha256": digest,
    }
    paths = ensure_dirs(config)
    raw_path = preseed

    result = run_acquire(config, paths, force=True)
    assert result["status"] == "completed"
    assert raw_path.is_file()
    assert sha256_file(raw_path) == digest


def test_preseed_hash_mismatch_triggers_network_block(
    tmp_path: Path, repo_root: Path, monkeypatch
) -> None:
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("not,the,preseed\n", encoding="utf-8")

    config = load_config("standard", repo_root=repo_root)
    config.raw["use_fixture"] = False
    config.raw["preseed_raw"] = {
        "path": str(wrong),
        "sha256": PRESEED_SHA256,
    }
    paths = ensure_dirs(config)
    monkeypatch.setenv("CRASHLAB_ALLOW_NETWORK", "0")

    with pytest.raises(Exception, match="Network acquisition disabled"):
        run_acquire(config, paths, force=True)


def test_smoke_fixture_acquire_offline(repo_root: Path, monkeypatch) -> None:
    monkeypatch.setenv("CRASHLAB_ALLOW_NETWORK", "0")
    config = load_config("smoke", repo_root=repo_root)
    paths = ensure_dirs(config)
    preseed_path = default_raw_path(paths, config)
    preseed_digest_before = sha256_file(preseed_path) if preseed_path.is_file() else None
    preseed_mtime_before = preseed_path.stat().st_mtime if preseed_path.is_file() else None

    expected_raw = fixture_acquire_path(paths, config)
    result = run_acquire(config, paths, force=True)

    assert result["status"] == "completed"
    assert result["row_count"] >= 20
    assert Path(result["raw_path"]).resolve() == expected_raw.resolve()
    assert expected_raw.is_file()
    assert expected_raw.name == "fixture_smoke.csv"

    if preseed_digest_before is not None:
        assert preseed_path.is_file()
        assert sha256_file(preseed_path) == preseed_digest_before
        assert preseed_path.stat().st_mtime == preseed_mtime_before


def test_fixture_acquire_refuses_preseed_path(tmp_path: Path, repo_root: Path) -> None:
    preseed = tmp_path / "brisbane_crashes_2015_2023.csv"
    preseed.write_text("crash_ref_number,crash_severity\n1,Minor Injury\n", encoding="utf-8")

    config = load_config("smoke", repo_root=repo_root)
    config.raw["fixture_raw_path"] = str(preseed)
    paths = ensure_dirs(config)

    with pytest.raises(AcquisitionError, match="must not write to preseed path"):
        run_acquire(config, paths, force=True)


def test_fixture_acquire_refuses_matching_preseed_sha(
    tmp_path: Path, repo_root: Path, monkeypatch
) -> None:
    fixture_src = repo_root / "data" / "samples" / "fixture.csv"
    assert fixture_src.is_file()

    preseed_like = tmp_path / "smoke_raw.csv"
    shutil.copy2(fixture_src, preseed_like)
    digest = sha256_file(preseed_like)

    config = load_config("smoke", repo_root=repo_root)
    config.raw["preseed_raw"] = {"path": str(preseed_like), "sha256": digest}
    config.raw["fixture_raw_path"] = str(preseed_like)
    paths = ensure_dirs(config)
    monkeypatch.setenv("CRASHLAB_ALLOW_NETWORK", "0")

    with pytest.raises(AcquisitionError, match="Refusing to overwrite preseed"):
        run_acquire(config, paths, force=True)
