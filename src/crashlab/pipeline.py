"""Minimal pipeline orchestration for bootstrap and smoke runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from crashlab.config import CrashlabConfig
from crashlab.data.acquire import run_acquire
from crashlab.data.clean import run_prepare
from crashlab.data.manifest import write_run_manifest
from crashlab.data.validate import run_validate
from crashlab.logging import get_logger
from crashlab.models.anomalies import run_anomaly_detection
from crashlab.models.binary import run_binary_training
from crashlab.models.hotspots import run_hotspot_clustering
from crashlab.models.multiclass import run_multiclass_training
from crashlab.models.ordinal import ordinal_enabled, run_ordinal_training
from crashlab.paths import CrashlabPaths, ensure_dirs

logger = get_logger("pipeline")

PIPELINE_STAGES: tuple[str, ...] = (
    "acquire",
    "validate",
    "prepare",
    "train-binary",
    "train-multiclass",
    "train-ordinal",
    "detect-anomalies",
    "cluster-hotspots",
    "train-counts",
    "report",
)

IMPLEMENTED_STAGES: frozenset[str] = frozenset(
    {
        "acquire",
        "validate",
        "prepare",
        "train-binary",
        "train-multiclass",
        "train-ordinal",
        "detect-anomalies",
        "cluster-hotspots",
    }
)

_STAGE_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "acquire": run_acquire,
    "validate": run_validate,
    "prepare": run_prepare,
    "train-binary": run_binary_training,
    "train-multiclass": run_multiclass_training,
    "train-ordinal": run_ordinal_training,
    "detect-anomalies": run_anomaly_detection,
    "cluster-hotspots": run_hotspot_clustering,
}


def run_stage(
    stage: str,
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run a single pipeline stage when implemented."""
    if stage not in _STAGE_RUNNERS:
        msg = f"Stage '{stage}' is not implemented yet (Phase D+)."
        raise NotImplementedError(msg)
    runner = _STAGE_RUNNERS[stage]
    return runner(config, paths, force=force)


def run_all(config: CrashlabConfig, *, force: bool = False) -> dict[str, float]:
    """Run implemented pipeline stages for the selected profile."""
    started = time.perf_counter()
    paths = ensure_dirs(config)
    logger.info("Profile loaded: %s (config hash %s)", config.profile, config.digest[:12])
    logger.info("Repository root: %s", config.repo_root)
    if config.use_fixture:
        logger.info("Fixture mode enabled: %s", config.fixture_path)

    timings: dict[str, float] = {}
    stage_results: dict[str, Any] = {}

    for index, stage in enumerate(PIPELINE_STAGES, start=1):
        if stage not in IMPLEMENTED_STAGES:
            logger.info("  %d. %s — pending implementation", index, stage)
            continue
        if stage == "train-ordinal" and not ordinal_enabled(config):
            logger.info("  %d. %s — skipped (disabled for profile)", index, stage)
            continue
        logger.info("  %d. Running %s", index, stage)
        stage_started = time.perf_counter()
        result = run_stage(stage, config, paths, force=force)
        elapsed = time.perf_counter() - stage_started
        timings[f"{stage}_seconds"] = elapsed
        stage_results[stage] = result
        logger.info("  %s finished in %.2fs (%s)", stage, elapsed, result.get("status"))

    elapsed = time.perf_counter() - started
    timings["total_seconds"] = elapsed
    manifest_path = _run_manifest_path(paths.manifests_dir, config.profile)
    write_run_manifest(
        manifest_path,
        config,
        command="all",
        status="completed_implemented_stages",
        timings=timings,
        extra={
            "stages": list(PIPELINE_STAGES),
            "implemented": sorted(IMPLEMENTED_STAGES),
            "results": stage_results,
        },
    )
    logger.info("Pipeline manifest written to %s", manifest_path)
    logger.info("Completed implemented stages in %.2fs", elapsed)
    return timings


def _run_manifest_path(manifests_dir: Path, profile: str) -> Path:
    return manifests_dir / f"run_all_{profile}.json"
