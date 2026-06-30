"""Cleaning, rejection tracking, and Parquet preparation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from crashlab.config import CrashlabConfig
from crashlab.data.acquire import resolve_raw_input_path
from crashlab.data.artifacts import processed_path, rejected_path
from crashlab.data.manifest import utc_now_iso
from crashlab.data.schema import (
    BRISBANE_LGA,
    COUNT_COLUMNS,
    HOUR_MAX,
    HOUR_MIN,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    MONTH_NAME_TO_INT,
    NULL_SENTINELS,
    SEVERITY_PDO,
    SEVERITY_VALUES,
    normalize_column_name,
    normalize_severity,
)
from crashlab.logging import get_logger
from crashlab.paths import CrashlabPaths

logger = get_logger("data.clean")

QUALITY_MANIFEST_NAME = "data_quality"
SUMMARY_NAME = "data_quality_summary.md"


def quality_json_path(paths: CrashlabPaths, profile: str) -> Path:
    return paths.manifests_dir / f"{QUALITY_MANIFEST_NAME}_{profile}.json"


def quality_summary_path(paths: CrashlabPaths, profile: str) -> Path:
    return paths.reports_dir / f"{SUMMARY_NAME.replace('.md', '')}_{profile}.md"


def _is_null(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    text = str(value).strip().lower()
    return text in NULL_SENTINELS


def _parse_year(value: object) -> int | None:
    if _is_null(value):
        return None
    try:
        year = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return year


def _parse_hour(value: object) -> int | None:
    if _is_null(value):
        return None
    try:
        hour = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return hour


def _parse_month(value: object) -> int | None:
    if _is_null(value):
        return None
    text = str(value).strip()
    if text.isdigit():
        month = int(text)
        return month if 1 <= month <= 12 else None
    return MONTH_NAME_TO_INT.get(text.lower())


def _parse_float(value: object) -> float | None:
    if _is_null(value):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_count(value: object) -> int | None:
    if _is_null(value):
        return 0
    try:
        count = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return count


def _parse_speed_limit(value: object) -> int | None:
    if _is_null(value):
        return None
    text = str(value).strip().lower().replace("km/h", "").replace("kmh", "").strip()
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def normalize_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and coerce core types."""
    renamed = {col: normalize_column_name(col) for col in df.columns}
    out = df.rename(columns=renamed).copy()

    if "crash_severity" in out.columns:
        out["crash_severity"] = out["crash_severity"].map(normalize_severity)

    for col in ("crash_year",):
        if col in out.columns:
            out[col] = out[col].map(_parse_year)

    if "crash_month" in out.columns:
        out["crash_month"] = out["crash_month"].map(_parse_month)

    if "crash_hour" in out.columns:
        out["crash_hour"] = out["crash_hour"].map(_parse_hour)

    for col in ("crash_longitude", "crash_latitude"):
        if col in out.columns:
            out[col] = out[col].map(_parse_float)

    if "crash_speed_limit" in out.columns:
        out["crash_speed_limit"] = out["crash_speed_limit"].map(_parse_speed_limit)

    for col in COUNT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(_parse_count)

    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].map(lambda value: None if _is_null(value) else str(value).strip())

    return out


def assign_rejection_reasons(
    df: pd.DataFrame,
    *,
    year_start: int,
    year_end: int,
    lga: str = BRISBANE_LGA,
) -> pd.Series:
    """Return semicolon-separated rejection reason codes per row (empty = keep)."""
    reasons: list[str] = [""] * len(df)

    def add_mask(mask: pd.Series, code: str) -> None:
        for pos in range(len(df)):
            if bool(mask.iloc[pos]):
                existing = reasons[pos]
                reasons[pos] = code if not existing else f"{existing};{code}"

    if "crash_severity" in df.columns:
        pdo = df["crash_severity"] == SEVERITY_PDO
        add_mask(pdo, "PDO_EXCLUDED")
        invalid_sev = df["crash_severity"].notna() & ~df["crash_severity"].isin(
            list(SEVERITY_VALUES) + [SEVERITY_PDO]
        )
        add_mask(invalid_sev, "INVALID_SEVERITY")
        missing_sev = df["crash_severity"].isna()
        add_mask(missing_sev, "INVALID_SEVERITY")

    if "crash_year" in df.columns:
        invalid_year = (
            df["crash_year"].isna()
            | (df["crash_year"] < year_start)
            | (df["crash_year"] > year_end)
        )
        add_mask(invalid_year, "INVALID_YEAR")

    if "crash_hour" in df.columns:
        invalid_hour = (
            df["crash_hour"].isna() | (df["crash_hour"] < HOUR_MIN) | (df["crash_hour"] > HOUR_MAX)
        )
        add_mask(invalid_hour, "INVALID_HOUR")

    if "loc_local_government_area" in df.columns:
        invalid_lga = df["loc_local_government_area"].fillna("").ne(lga)
        add_mask(invalid_lga, "INVALID_LGA")

    if "crash_latitude" in df.columns and "crash_longitude" in df.columns:
        lat = df["crash_latitude"]
        lon = df["crash_longitude"]
        invalid_coords = (
            lat.isna()
            | lon.isna()
            | (lat < LAT_MIN)
            | (lat > LAT_MAX)
            | (lon < LON_MIN)
            | (lon > LON_MAX)
        )
        add_mask(invalid_coords, "INVALID_COORDINATES")

    for col in COUNT_COLUMNS:
        if col not in df.columns:
            continue
        invalid_count = df[col].isna() | (df[col] < 0)
        add_mask(invalid_count, "INVALID_COUNT")

    return pd.Series(reasons, index=df.index)


def split_clean_rejected(
    df: pd.DataFrame,
    *,
    year_start: int,
    year_end: int,
    lga: str = BRISBANE_LGA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition into modelling-ready rows and rejected rows with reasons."""
    working = normalize_raw_frame(df)
    working = working.copy()
    working["_rejection_reason"] = assign_rejection_reasons(
        working,
        year_start=year_start,
        year_end=year_end,
        lga=lga,
    )

    duplicate_mask = working.duplicated(keep="first")
    dup_indices = working.index[duplicate_mask]
    working.loc[dup_indices, "_rejection_reason"] = working.loc[
        dup_indices, "_rejection_reason"
    ].map(lambda existing: "DUPLICATE_ROW" if not existing else f"{existing};DUPLICATE_ROW")

    rejected = working[working["_rejection_reason"].astype(bool)].copy()
    clean = working[~working["_rejection_reason"].astype(bool)].copy()

    if "crash_ref_number" in clean.columns:
        ref_dupes = clean.duplicated(subset=["crash_ref_number"], keep=False)
        clean.loc[ref_dupes, "duplicate_ref_flag"] = True
        clean.loc[~ref_dupes, "duplicate_ref_flag"] = False
    else:
        clean["duplicate_ref_flag"] = False

    for frame in (clean, rejected):
        if "_rejection_reason" in frame.columns and "rejection_reason" not in frame.columns:
            frame.rename(columns={"_rejection_reason": "rejection_reason"}, inplace=True)

    return clean, rejected


def build_quality_report(
    raw_count: int,
    clean: pd.DataFrame,
    rejected: pd.DataFrame,
    *,
    profile: str,
) -> dict[str, Any]:
    coord_valid = 0.0
    if len(clean) and {"crash_latitude", "crash_longitude"}.issubset(clean.columns):
        lat = clean["crash_latitude"]
        lon = clean["crash_longitude"]
        valid = (
            lat.notna()
            & lon.notna()
            & (lat >= LAT_MIN)
            & (lat <= LAT_MAX)
            & (lon >= LON_MIN)
            & (lon <= LON_MAX)
        )
        coord_valid = float(valid.mean())

    reason_counts: dict[str, int] = {}
    if "rejection_reason" in rejected.columns:
        for cell in rejected["rejection_reason"].fillna(""):
            for code in str(cell).split(";"):
                if code:
                    reason_counts[code] = reason_counts.get(code, 0) + 1

    missingness = {
        col: float(clean[col].isna().mean()) if col in clean.columns else 1.0
        for col in sorted(clean.columns)
    }
    cardinality = {
        col: int(clean[col].nunique(dropna=True)) if col in clean.columns else 0
        for col in sorted(clean.columns)
    }

    rejection_rate = len(rejected) / raw_count if raw_count else 0.0
    return {
        "schema_version": "1",
        "manifest_type": "data_quality",
        "timestamp_utc": utc_now_iso(),
        "profile": profile,
        "raw_row_count": raw_count,
        "clean_row_count": len(clean),
        "rejected_row_count": len(rejected),
        "rejection_rate": rejection_rate,
        "coordinate_valid_rate": coord_valid,
        "rejection_reason_counts": reason_counts,
        "missingness": missingness,
        "cardinality": cardinality,
        "severity_counts": (
            clean["crash_severity"].value_counts(dropna=False).to_dict()
            if "crash_severity" in clean.columns
            else {}
        ),
    }


def render_quality_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Data quality summary",
        "",
        f"- Profile: `{report.get('profile', 'unknown')}`",
        f"- Raw rows: {report.get('raw_row_count', 0)}",
        f"- Clean rows: {report.get('clean_row_count', 0)}",
        f"- Rejected rows: {report.get('rejected_row_count', 0)}",
        f"- Rejection rate: {report.get('rejection_rate', 0.0):.2%}",
        f"- Coordinate-valid rate (clean): {report.get('coordinate_valid_rate', 0.0):.2%}",
        "",
        "## Rejection reasons",
        "",
    ]
    for code, count in sorted(report.get("rejection_reason_counts", {}).items()):
        lines.append(f"- {code}: {count}")
    lines.extend(["", "## Severity distribution (clean)", ""])
    for severity, count in sorted(
        report.get("severity_counts", {}).items(), key=lambda item: str(item[0])
    ):
        lines.append(f"- {severity}: {count}")
    return "\n".join(lines) + "\n"


def run_prepare(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Clean raw data and write Parquet outputs plus quality artifacts."""
    started = time.perf_counter()
    raw_path = resolve_raw_input_path(config, paths)
    out_parquet = processed_path(paths)
    out_rejected = rejected_path(paths)
    quality_json = quality_json_path(paths, config.profile)
    quality_md = quality_summary_path(paths, config.profile)

    feature_manifest = paths.manifests_dir / f"features_{config.profile}.json"
    eda_manifest = paths.manifests_dir / f"eda_{config.profile}.json"
    if (
        not force
        and out_parquet.is_file()
        and quality_json.is_file()
        and feature_manifest.is_file()
        and eda_manifest.is_file()
    ):
        logger.info("Prepared outputs exist; skipping (use --force to re-run)")
        return {
            "status": "skipped",
            "processed_path": str(out_parquet),
            "quality_manifest": str(quality_json),
            "features_manifest": str(feature_manifest),
            "eda_manifest": str(eda_manifest),
        }

    if not raw_path.is_file():
        msg = f"Raw input not found for prepare: {raw_path}"
        raise FileNotFoundError(msg)

    raw_df = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    year_start = int(config.data.get("year_start", 2015))
    year_end = int(config.data.get("year_end", 2023))
    lga = str(config.data.get("lga", BRISBANE_LGA))

    clean, rejected = split_clean_rejected(
        raw_df,
        year_start=year_start,
        year_end=year_end,
        lga=lga,
    )

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_parquet, compression="snappy", index=False)
    if len(rejected):
        rejected.to_parquet(out_rejected, compression="snappy", index=False)
    elif out_rejected.exists():
        out_rejected.unlink()

    report = build_quality_report(len(raw_df), clean, rejected, profile=config.profile)
    quality_json.parent.mkdir(parents=True, exist_ok=True)
    with quality_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    quality_md.parent.mkdir(parents=True, exist_ok=True)
    quality_md.write_text(render_quality_summary(report), encoding="utf-8")

    from crashlab.evaluation.eda import run_eda
    from crashlab.features.build import run_feature_build

    feature_result = run_feature_build(config, paths, force=force, df=clean)
    eda_result = run_eda(config, paths, force=force, df=clean)

    elapsed = time.perf_counter() - started
    logger.info(
        "Prepare complete: %d clean / %d rejected rows -> %s (%.2fs)",
        len(clean),
        len(rejected),
        out_parquet,
        elapsed,
    )
    return {
        "status": "completed",
        "processed_path": str(out_parquet),
        "rejected_path": str(out_rejected) if len(rejected) else None,
        "quality_manifest": str(quality_json),
        "quality_summary": str(quality_md),
        "clean_row_count": len(clean),
        "rejected_row_count": len(rejected),
        "features": feature_result,
        "eda": eda_result,
        "timings": {"prepare_seconds": elapsed},
    }
