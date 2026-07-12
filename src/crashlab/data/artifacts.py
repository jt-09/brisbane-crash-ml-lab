"""Shared artifact path helpers for data pipeline outputs."""

from __future__ import annotations

from pathlib import Path

from crashlab.paths import CrashlabPaths

PROCESSED_NAME = "brisbane_crashes_cleaned_{profile}.parquet"
REJECTED_NAME = "brisbane_crashes_rejected.parquet"


def processed_path(paths: CrashlabPaths, profile: str) -> Path:
    return paths.processed_dir / PROCESSED_NAME.format(profile=profile)


def rejected_path(paths: CrashlabPaths) -> Path:
    return paths.interim_dir / REJECTED_NAME
