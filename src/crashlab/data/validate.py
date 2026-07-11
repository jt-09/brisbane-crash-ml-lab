"""Raw data contract validation before cleaning."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from crashlab.config import CrashlabConfig
from crashlab.data.acquire import resolve_raw_input_path
from crashlab.data.manifest import utc_now_iso
from crashlab.data.schema import (
    BRISBANE_LGA,
    REQUIRED_FIELDS,
    normalize_column_name,
    normalize_severity,
)
from crashlab.logging import get_logger
from crashlab.paths import CrashlabPaths

logger = get_logger("data.validate")


class ValidationError(RuntimeError):
    """Raised when raw data fails contract checks."""


def validation_manifest_path(paths: CrashlabPaths, profile: str) -> Path:
    return paths.manifests_dir / f"validation_{profile}.json"


def _load_raw_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"Raw data file not found: {path}"
        raise ValidationError(msg)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: normalize_column_name(col) for col in df.columns}
    return df.rename(columns=renamed)


def validate_raw_frame(
    df: pd.DataFrame,
    *,
    year_start: int,
    year_end: int,
    lga: str = BRISBANE_LGA,
) -> dict[str, Any]:
    """Run schema and content checks on a normalised raw frame."""
    issues: list[str] = []
    columns = set(df.columns)

    missing_required = [field for field in REQUIRED_FIELDS if field not in columns]
    if missing_required:
        issues.append(f"missing_required:{','.join(missing_required)}")

    row_count = len(df)
    if row_count == 0:
        issues.append("empty_dataset")

    if "crash_severity" in df.columns:
        severities = {
            str(value).strip() for value in df["crash_severity"].unique() if str(value).strip()
        }
        unknown = sorted(sev for sev in severities if normalize_severity(sev) is None)
        if unknown:
            issues.append(f"unknown_severity:{','.join(unknown[:10])}")

    if "loc_local_government_area" in df.columns and row_count:
        lga_values = {
            str(value).strip()
            for value in df["loc_local_government_area"].unique()
            if str(value).strip()
        }
        non_target = sorted(value for value in lga_values if value != lga)
        if non_target:
            issues.append(f"unexpected_lga:{','.join(non_target[:5])}")

    if "crash_year" in df.columns and row_count:
        years = pd.to_numeric(df["crash_year"], errors="coerce")
        valid_years = years.dropna()
        if not valid_years.empty:
            ymin = int(valid_years.min())
            ymax = int(valid_years.max())
            if ymin < year_start or ymax > year_end:
                issues.append(f"year_out_of_range:{ymin}-{ymax}")

    passed = not issues
    return {
        "passed": passed,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": sorted(columns),
        "missing_required": missing_required,
        "issues": issues,
        "year_start": year_start,
        "year_end": year_end,
        "lga": lga,
    }


def run_validate(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Validate raw CSV against the project data contract."""
    started = time.perf_counter()
    raw_path = resolve_raw_input_path(config, paths)
    manifest_path = validation_manifest_path(paths, config.profile)

    if not force and manifest_path.is_file():
        logger.info("Validation manifest exists; skipping (use --force to re-run)")
        return {"status": "skipped", "manifest_path": str(manifest_path)}

    df = _load_raw_csv(raw_path)
    df = _normalize_columns(df)
    year_start = int(config.data.get("year_start", 2015))
    year_end = int(config.data.get("year_end", 2023))
    lga = str(config.data.get("lga", BRISBANE_LGA))

    report = validate_raw_frame(
        df,
        year_start=year_start,
        year_end=year_end,
        lga=lga if not config.use_fixture else lga,
    )
    elapsed = time.perf_counter() - started

    payload: dict[str, Any] = {
        "schema_version": "1",
        "manifest_type": "validation",
        "timestamp_utc": utc_now_iso(),
        "profile": config.profile,
        "config_hash": config.digest,
        "raw_path": str(raw_path),
        "status": "passed" if report["passed"] else "failed",
        "report": report,
        "timings_seconds": {"validate_seconds": elapsed},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if not report["passed"]:
        msg = f"Raw validation failed: {report['issues']}"
        raise ValidationError(msg)

    logger.info("Validation passed for %s (%d rows)", raw_path, report["row_count"])
    return {
        "status": "completed",
        "manifest_path": str(manifest_path),
        "report": report,
        "timings": {"validate_seconds": elapsed},
    }
