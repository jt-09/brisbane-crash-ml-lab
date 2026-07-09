"""Load precomputed metrics and model artifacts for reporting and the app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crashlab.config import CrashlabConfig
from crashlab.data.clean import quality_json_path, quality_summary_path
from crashlab.models.common import artifacts_metrics_dir, metrics_artifact_path
from crashlab.paths import CrashlabPaths


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def load_binary_metrics(config: CrashlabConfig, paths: CrashlabPaths) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for moment in ("context", "triage", "leakage_demo"):
        path = metrics_artifact_path(paths, "binary", config.profile, moment)
        payload = _read_json(path)
        if payload:
            out[moment] = payload
    return out


def load_task_metrics(config: CrashlabConfig, paths: CrashlabPaths) -> dict[str, Any]:
    """Aggregate metrics JSON artifacts for all modelling tasks."""
    metrics_dir = artifacts_metrics_dir(paths)
    profile = config.profile
    return {
        "binary": load_binary_metrics(config, paths),
        "multiclass": {
            m: _read_json(metrics_artifact_path(paths, "multiclass", profile, m))
            for m in ("context", "triage")
        },
        "ordinal": {
            m: _read_json(metrics_artifact_path(paths, "ordinal", profile, m))
            for m in ("context", "triage")
        },
        "anomalies": _read_json(metrics_dir / f"anomalies_{profile}.json"),
        "hotspots": _read_json(metrics_dir / f"hotspots_{profile}.json"),
        "counts": _read_json(metrics_dir / f"counts_{profile}.json"),
        "explanation": _read_json(metrics_dir / f"explanation_{profile}.json"),
        "eda": _read_json(paths.manifests_dir / f"eda_{profile}.json"),
        "data_quality": _read_json(quality_json_path(paths, profile)),
        "run_all": _read_json(paths.manifests_dir / f"run_all_{profile}.json"),
    }


def quality_summary_markdown(config: CrashlabConfig, paths: CrashlabPaths) -> str | None:
    path = quality_summary_path(paths, config.profile)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def list_figure_paths(paths: CrashlabPaths) -> list[Path]:
    if not paths.figures_dir.is_dir():
        return []
    return sorted(paths.figures_dir.glob("*.png"))
