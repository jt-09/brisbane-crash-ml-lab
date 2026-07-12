"""Feature matrix construction for context, triage, and leakage-demo moments."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib  # type: ignore[import-untyped]
import pandas as pd

from crashlab.config import CrashlabConfig
from crashlab.data.artifacts import processed_path
from crashlab.data.manifest import utc_now_iso
from crashlab.features.constants import (
    CONTEXT_CATEGORICAL,
    CONTEXT_EXCLUDED,
    DERIVED_NUMERIC,
    FEATURE_MOMENTS,
    IDENTIFIER_DENYLIST,
    LEAKAGE_DEMO_EXTRA,
    LEAKAGE_DENYLIST,
    LEAKAGE_DENYLIST_PATTERNS,
    TRIAGE_EXTRA_CATEGORICAL,
    TRIAGE_EXTRA_NUMERIC,
)
from crashlab.features.encoders import (
    EncoderBundle,
    fit_encoder_bundle,
    transform_with_mixed_encoding,
)
from crashlab.features.spatial import add_spatial_cell
from crashlab.features.targets import (
    add_binary_target,
    add_multiclass_target,
    filter_modeling_rows,
)
from crashlab.features.temporal import (
    YearSplits,
    add_cyclic_time_features,
    add_speed_bucket,
    assign_split_column,
    compute_year_splits,
)
from crashlab.logging import get_logger
from crashlab.paths import CrashlabPaths

logger = get_logger("features.build")

FeatureMoment = Literal["context", "triage", "leakage_demo"]

ENCODER_NAME = "feature_encoders_{profile}_{moment}.joblib"
MATRIX_NAME = "features_{moment}_{split}.parquet"
MANIFEST_NAME = "features_{profile}.json"


@dataclass
class FeatureBuildResult:
    moment: str
    splits: dict[str, str]
    feature_names: list[str]
    row_counts: dict[str, int]


def _contains_leakage(feature_names: list[str]) -> list[str]:
    """Return denylisted names present in a feature column list."""
    violations: list[str] = []
    lowered = {name.lower() for name in feature_names}
    for field in LEAKAGE_DENYLIST:
        if field in lowered:
            violations.append(field)
    for name in feature_names:
        low = name.lower()
        if any(pattern in low for pattern in LEAKAGE_DENYLIST_PATTERNS):
            violations.append(name)
    return sorted(set(violations))


def assert_valid_feature_columns(feature_names: list[str], moment: FeatureMoment) -> None:
    """Raise when denylisted fields appear in a valid severity feature pipeline."""
    if moment == "leakage_demo":
        return
    violations = _contains_leakage(feature_names)
    if violations:
        msg = f"Leakage denylist violation in {moment} features: {violations}"
        raise ValueError(msg)
    if moment == "context":
        for name in feature_names:
            base = name.split("__")[0]
            if base in CONTEXT_EXCLUDED:
                msg = f"Context moment includes excluded column: {base}"
                raise ValueError(msg)


def _add_vru_context_flag(df: pd.DataFrame) -> pd.DataFrame:
    """VRU proxy from crash type text when unit counts are unavailable."""
    out = df.copy()
    if "crash_type" in out.columns:
        out["is_vru_context"] = (
            out["crash_type"].astype(str).str.contains("Pedestrian|Bicycle", case=False, na=False)
        ).astype(int)
    else:
        out["is_vru_context"] = 0
    return out


def _add_vru_triage_flag(df: pd.DataFrame) -> pd.DataFrame:
    """VRU indicator from vulnerable unit counts (triage moment only)."""
    out = df.copy()
    bike_series = (
        out["count_unit_bicycle"]
        if "count_unit_bicycle" in out.columns
        else pd.Series(0, index=out.index)
    )
    ped_series = (
        out["count_unit_pedestrian"]
        if "count_unit_pedestrian" in out.columns
        else pd.Series(0, index=out.index)
    )
    bike = pd.to_numeric(bike_series, errors="coerce").fillna(0)
    ped = pd.to_numeric(ped_series, errors="coerce").fillna(0)
    out["is_vru_triage"] = ((bike > 0) | (ped > 0)).astype(int)
    return out


def prepare_derived_frame(df: pd.DataFrame, moment: FeatureMoment) -> pd.DataFrame:
    """Apply shared derived-feature transforms before encoding."""
    out = filter_modeling_rows(df)
    out = add_cyclic_time_features(out)
    out = add_speed_bucket(out)
    out = add_spatial_cell(out)
    out = _add_vru_context_flag(out)
    if moment == "triage":
        out = _add_vru_triage_flag(out)
    return out


def _columns_for_moment(moment: FeatureMoment) -> tuple[list[str], list[str]]:
    categorical = list(CONTEXT_CATEGORICAL)
    numeric = list(DERIVED_NUMERIC)
    if moment == "triage":
        categorical.extend(TRIAGE_EXTRA_CATEGORICAL)
        numeric.extend(TRIAGE_EXTRA_NUMERIC)
        numeric.append("is_vru_triage")
    if moment == "leakage_demo":
        numeric.extend(LEAKAGE_DEMO_EXTRA)
    return categorical, numeric


def _resolve_year_splits(config: CrashlabConfig, years: list[int]) -> YearSplits:
    splits_cfg = config.raw.get("splits", {})
    if not isinstance(splits_cfg, dict):
        splits_cfg = {}
    train_end = splits_cfg.get("train_year_end")
    val_years = splits_cfg.get("val_years")
    test_years = splits_cfg.get("test_years")
    return compute_year_splits(
        years,
        train_year_end=int(train_end) if train_end is not None else None,
        val_years=[int(y) for y in val_years] if isinstance(val_years, list) else None,
        test_years=[int(y) for y in test_years] if isinstance(test_years, list) else None,
    )


def features_dir(paths: CrashlabPaths, profile: str) -> Path:
    return paths.processed_dir / "features" / profile


def encoder_path(paths: CrashlabPaths, profile: str, moment: str) -> Path:
    return paths.interim_dir / ENCODER_NAME.format(profile=profile, moment=moment)


def matrix_path(paths: CrashlabPaths, profile: str, moment: str, split: str) -> Path:
    return features_dir(paths, profile) / MATRIX_NAME.format(moment=moment, split=split)


def manifest_path(paths: CrashlabPaths, profile: str) -> Path:
    return paths.manifests_dir / MANIFEST_NAME.format(profile=profile)


def build_moment_features(
    df: pd.DataFrame,
    moment: FeatureMoment,
    splits: YearSplits,
    *,
    min_count: int = 2,
) -> tuple[dict[str, pd.DataFrame], EncoderBundle, list[str]]:
    """Build feature matrices for each split of one prediction moment."""
    derived = prepare_derived_frame(df, moment)
    derived["split"] = assign_split_column(derived, splits)

    train_mask = derived["split"] == "train"
    train_df = derived.loc[train_mask]
    if train_df.empty:
        msg = f"No training rows for moment {moment}"
        raise ValueError(msg)

    cat_cols, num_cols = _columns_for_moment(moment)
    cat_cols = [c for c in cat_cols if c in derived.columns]
    num_cols = [c for c in num_cols if c in derived.columns]

    bundle = fit_encoder_bundle(
        train_df,
        cat_cols,
        num_cols,
        min_count=min_count,
        encoding="one_hot",
    )

    matrices: dict[str, pd.DataFrame] = {}
    feature_names: list[str] = []
    for split_name in ("train", "val", "test"):
        split_df = derived.loc[derived["split"] == split_name]
        if split_df.empty:
            continue
        features = transform_with_mixed_encoding(bundle, split_df)
        feature_names = list(features.columns)
        assert_valid_feature_columns(feature_names, moment)
        for col in IDENTIFIER_DENYLIST:
            if col in features.columns:
                msg = f"Identifier column {col} in {moment} features"
                raise ValueError(msg)
        matrices[split_name] = features

    return matrices, bundle, feature_names


def run_feature_build(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    force: bool = False,
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build leakage-safe feature matrices and persist artifacts."""
    started = time.perf_counter()
    manifest = manifest_path(paths, config.profile)
    if not force and manifest.is_file():
        logger.info("Feature artifacts exist; skipping (use --force)")
        return {"status": "skipped", "manifest": str(manifest)}

    if df is None:
        parquet = processed_path(paths, config.profile)
        if not parquet.is_file():
            msg = f"Cleaned parquet required before feature build: {parquet}"
            raise FileNotFoundError(msg)
        df = pd.read_parquet(parquet)

    modeling = filter_modeling_rows(df)
    modeling = add_binary_target(modeling)
    modeling = add_multiclass_target(modeling)

    years = sorted(int(y) for y in modeling["crash_year"].dropna().unique())
    year_splits = _resolve_year_splits(config, years)
    year_splits.validate_disjoint()

    features_dir(paths, config.profile).mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"moments": {}, "splits": year_splits.as_dict()}

    for moment in FEATURE_MOMENTS:
        matrices, bundle, feature_names = build_moment_features(
            modeling,
            moment,  # type: ignore[arg-type]
            year_splits,
            min_count=2 if config.profile == "smoke" else 5,
        )
        joblib.dump(bundle, encoder_path(paths, config.profile, moment))
        row_counts: dict[str, int] = {}
        for split_name, matrix in matrices.items():
            out_path = matrix_path(paths, config.profile, moment, split_name)
            matrix.to_parquet(out_path, compression="snappy", index=False)
            row_counts[split_name] = len(matrix)
        results["moments"][moment] = {
            "feature_names": feature_names,
            "row_counts": row_counts,
            "encoder_path": str(encoder_path(paths, config.profile, moment)),
        }
        logger.info(
            "Built %s features: %d columns, splits %s",
            moment,
            len(feature_names),
            row_counts,
        )

    elapsed = time.perf_counter() - started
    manifest_payload = {
        "schema_version": "1",
        "manifest_type": "features",
        "timestamp_utc": utc_now_iso(),
        "profile": config.profile,
        "splits": year_splits.as_dict(),
        "moments": results["moments"],
        "timings": {"feature_build_seconds": elapsed},
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return {
        "status": "completed",
        "manifest": str(manifest),
        "splits": year_splits.as_dict(),
        "moments": list(FEATURE_MOMENTS),
        "timings": {"feature_build_seconds": elapsed},
    }


def get_feature_names(
    moment: FeatureMoment, config: CrashlabConfig, paths: CrashlabPaths
) -> list[str]:
    """Load feature names from the persisted manifest."""
    path = manifest_path(paths, config.profile)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    moment_data = data.get("moments", {}).get(moment, {})
    names = moment_data.get("feature_names", [])
    return list(names) if isinstance(names, list) else []
