"""Minimal pipeline orchestration for bootstrap and smoke runs."""

from __future__ import annotations

import time
from pathlib import Path

from crashlab.config import CrashlabConfig
from crashlab.data.manifest import write_run_manifest
from crashlab.logging import get_logger
from crashlab.paths import ensure_dirs

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


def run_all(config: CrashlabConfig, *, force: bool = False) -> dict[str, float]:
    """Bootstrap ``all`` command: wire config, logging, dirs, and stage plan."""
    del force  # reserved for later phases
    started = time.perf_counter()
    paths = ensure_dirs(config)
    logger.info("Profile loaded: %s (config hash %s)", config.profile, config.digest[:12])
    logger.info("Repository root: %s", config.repo_root)
    if config.use_fixture:
        logger.info("Fixture mode enabled: %s", config.fixture_path)
    else:
        logger.info("Fixture mode disabled — real data paths will be used in later phases")

    logger.info("Planned pipeline stages (Phase B+ not yet implemented):")
    for index, stage in enumerate(PIPELINE_STAGES, start=1):
        logger.info("  %d. %s — pending implementation", index, stage)

    elapsed = time.perf_counter() - started
    manifest_path = _bootstrap_manifest_path(paths.manifests_dir, config.profile)
    write_run_manifest(
        manifest_path,
        config,
        command="all",
        status="bootstrap_plan_only",
        timings={"bootstrap_seconds": elapsed},
        extra={"stages": list(PIPELINE_STAGES), "implemented": False},
    )
    logger.info("Bootstrap manifest written to %s", manifest_path)
    logger.info("Smoke/bootstrap planning complete in %.2fs", elapsed)
    return {"bootstrap_seconds": elapsed}


def _bootstrap_manifest_path(manifests_dir: Path, profile: str) -> Path:
    return manifests_dir / f"run_all_{profile}_bootstrap.json"
