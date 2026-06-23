"""Run and acquisition manifest read/write helpers."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crashlab.config import CrashlabConfig, config_hash


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def get_git_sha(repo_root: Path | None = None) -> str | None:
    """Best-effort current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        msg = f"Manifest must be a JSON object: {path}"
        raise ValueError(msg)
    return data


def build_run_manifest(
    config: CrashlabConfig,
    *,
    command: str,
    status: str = "started",
    timings: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a run manifest payload."""
    manifest: dict[str, Any] = {
        "schema_version": "1",
        "manifest_type": "run",
        "timestamp_utc": utc_now_iso(),
        "git_sha": get_git_sha(config.repo_root),
        "profile": config.profile,
        "config_hash": config.digest,
        "command": command,
        "status": status,
        "timings_seconds": timings or {},
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_run_manifest(
    path: Path,
    config: CrashlabConfig,
    *,
    command: str,
    status: str = "started",
    timings: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_run_manifest(
        config,
        command=command,
        status=status,
        timings=timings,
        extra=extra,
    )
    _write_json(path, payload)
    return payload


def read_run_manifest(path: Path) -> dict[str, Any]:
    return _read_json(path)


def build_acquisition_manifest(
    *,
    config: CrashlabConfig | dict[str, Any],
    source_url: str,
    raw_path: Path,
    byte_size: int,
    row_count: int,
    sha256: str,
    filters: dict[str, Any] | None = None,
    selected_fields: list[str] | None = None,
    timings: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct an acquisition provenance manifest payload."""
    if isinstance(config, CrashlabConfig):
        profile = config.profile
        digest = config.digest
        repo_root = config.repo_root
    else:
        profile = str(config.get("profile", "unknown"))
        digest = config_hash(config)
        repo_root = None

    manifest: dict[str, Any] = {
        "schema_version": "1",
        "manifest_type": "acquisition",
        "timestamp_utc": utc_now_iso(),
        "git_sha": get_git_sha(repo_root),
        "profile": profile,
        "config_hash": digest,
        "source_url": source_url,
        "raw_path": str(raw_path),
        "byte_size": byte_size,
        "row_count": row_count,
        "sha256": sha256,
        "filters": filters or {},
        "selected_fields": selected_fields or [],
        "timings_seconds": timings or {},
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_acquisition_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, payload)
    return payload


def read_acquisition_manifest(path: Path) -> dict[str, Any]:
    return _read_json(path)
