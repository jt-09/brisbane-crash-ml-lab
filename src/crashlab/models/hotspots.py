"""Spatial clustering of recorded crash locations.

Clusters summarise where crashes were *recorded* in the dataset. They do not
measure intrinsic road danger or exposure-adjusted risk without traffic volume.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN  # type: ignore[import-untyped]

from crashlab.config import CrashlabConfig
from crashlab.data.artifacts import processed_path
from crashlab.data.manifest import utc_now_iso
from crashlab.data.schema import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN
from crashlab.evaluation.stability import cluster_stability_ari
from crashlab.features.spatial import DEFAULT_GRID_SIZE, add_spatial_cell
from crashlab.features.targets import add_binary_target, filter_modeling_rows
from crashlab.features.temporal import add_cyclic_time_features
from crashlab.logging import get_logger
from crashlab.models.common import artifacts_metrics_dir, persist_json, set_random_seed
from crashlab.paths import CrashlabPaths

logger = get_logger("models.hotspots")

SubsetName = Literal["all", "severe", "night", "wet", "motorcycle", "vru"]

EARTH_RADIUS_KM = 6371.0
DEFAULT_DBSCAN_EPS_METERS = 500.0
DEFAULT_DBSCAN_MIN_SAMPLES = 5


def _series_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


SUBSET_DEFINITIONS: dict[SubsetName, str] = {
    "all": "All casualty crashes with valid coordinates",
    "severe": "Fatal or hospitalised crashes",
    "night": "Hour < 6 or hour >= 20",
    "wet": "Wet sealed surface or raining conditions",
    "motorcycle": "At least one motorcycle/moped unit",
    "vru": "Bicycle or pedestrian unit involved",
}


def _configured_methods(config: CrashlabConfig) -> list[str]:
    methods = config.models.get("spatial", ["grid"])
    if not isinstance(methods, list):
        return ["grid"]
    return [str(m) for m in methods]


def _load_spatial_frame(paths: CrashlabPaths) -> pd.DataFrame:
    parquet = processed_path(paths)
    if not parquet.is_file():
        msg = f"Processed parquet required: {parquet}"
        raise FileNotFoundError(msg)
    df = pd.read_parquet(parquet)
    df = filter_modeling_rows(df)
    df = add_binary_target(add_cyclic_time_features(df))
    df = add_spatial_cell(df)
    return df.reset_index(drop=True)


def valid_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with coordinates inside the Brisbane bounding box."""
    lon = pd.to_numeric(df["crash_longitude"], errors="coerce")
    lat = pd.to_numeric(df["crash_latitude"], errors="coerce")
    mask = (
        lon.notna()
        & lat.notna()
        & (lon >= LON_MIN)
        & (lon <= LON_MAX)
        & (lat >= LAT_MIN)
        & (lat <= LAT_MAX)
    )
    return df.loc[mask].copy()


def filter_subset(df: pd.DataFrame, subset: SubsetName) -> pd.DataFrame:
    """Filter crash records for a named spatial subset when columns allow."""
    if subset == "all":
        return df
    if subset == "severe":
        if "severe_binary" not in df.columns:
            return df.iloc[0:0]
        return df.loc[df["severe_binary"] == 1]
    if subset == "night":
        hour = _series_column(df, "crash_hour")
        return df.loc[(hour < 6) | (hour >= 20)].copy()
    if subset == "wet":
        surface = df.get("crash_road_surface_condition", pd.Series("", index=df.index)).astype(str)
        atmos = df.get("crash_atmospheric_condition", pd.Series("", index=df.index)).astype(str)
        wet = surface.str.contains("wet", case=False, na=False) | atmos.str.contains(
            "rain", case=False, na=False
        )
        return df.loc[wet]
    if subset == "motorcycle":
        col = "count_unit_motorcycle_moped"
        if col not in df.columns:
            return df.iloc[0:0]
        units = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df.loc[units > 0]
    if subset == "vru":
        bike = pd.to_numeric(
            df.get("count_unit_bicycle", pd.Series(0, index=df.index)), errors="coerce"
        ).fillna(0)
        ped = pd.to_numeric(
            df.get("count_unit_pedestrian", pd.Series(0, index=df.index)), errors="coerce"
        ).fillna(0)
        return df.loc[(bike > 0) | (ped > 0)]
    return df


def grid_count_summary(df: pd.DataFrame, *, grid_size: float = DEFAULT_GRID_SIZE) -> dict[str, Any]:
    """Deterministic grid-cell crash counts (recorded-crash density only)."""
    framed = add_spatial_cell(df, grid_size=grid_size)
    counts = framed["spatial_cell"].value_counts()
    return {
        "method": "grid",
        "grid_size_degrees": grid_size,
        "n_points": len(framed),
        "n_cells": int(counts.shape[0]),
        "max_cell_count": int(counts.max()) if len(counts) else 0,
        "median_cell_count": float(counts.median()) if len(counts) else 0.0,
        "top_cells": counts.head(10).astype(int).to_dict(),
        "exposure_note": (
            "Grid counts reflect recorded crashes only; not exposure-adjusted road risk."
        ),
    }


def _coords_radians(df: pd.DataFrame) -> np.ndarray:
    lat = pd.to_numeric(df["crash_latitude"], errors="coerce").to_numpy()
    lon = pd.to_numeric(df["crash_longitude"], errors="coerce").to_numpy()
    return np.column_stack([np.radians(lat), np.radians(lon)])


def dbscan_cluster(
    df: pd.DataFrame,
    *,
    eps_meters: float = DEFAULT_DBSCAN_EPS_METERS,
    min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
) -> dict[str, Any]:
    """DBSCAN on radian coordinates with haversine distance."""
    if len(df) < min_samples:
        return {
            "method": "dbscan",
            "n_points": len(df),
            "skipped": True,
            "reason": "insufficient_points",
        }
    coords = _coords_radians(df)
    eps_rad = eps_meters / 1000.0 / EARTH_RADIUS_KM
    model = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    labels = model.fit_predict(coords)
    n_noise = int((labels == -1).sum())
    unique_clusters = sorted({int(x) for x in labels if x >= 0})
    cluster_sizes = {str(c): int((labels == c).sum()) for c in unique_clusters}
    return {
        "method": "dbscan",
        "n_points": len(df),
        "eps_meters": eps_meters,
        "eps_radians": eps_rad,
        "min_samples": min_samples,
        "n_clusters": len(unique_clusters),
        "noise_fraction": float(n_noise / len(df)) if len(df) else 0.0,
        "cluster_sizes": cluster_sizes,
        "exposure_note": (
            "DBSCAN clusters group recorded crash locations; not intrinsic danger rankings."
        ),
        "labels": labels.tolist(),
    }


def hdbscan_cluster(
    df: pd.DataFrame,
    *,
    min_cluster_size: int = 5,
) -> dict[str, Any]:
    """HDBSCAN when an implementation is available; otherwise report unavailable."""
    estimator: Any | None = None
    try:
        from sklearn.cluster import HDBSCAN  # type: ignore[attr-defined,import-untyped]

        estimator = HDBSCAN(min_cluster_size=min_cluster_size, metric="haversine")
    except ImportError:
        try:
            import hdbscan  # type: ignore[import-not-found,import-untyped]

            estimator = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric="haversine",
            )
        except ImportError:
            return {
                "method": "hdbscan",
                "available": False,
                "reason": "HDBSCAN not installed in this environment",
            }

    if len(df) < min_cluster_size:
        return {
            "method": "hdbscan",
            "available": True,
            "skipped": True,
            "reason": "insufficient_points",
        }

    coords = _coords_radians(df)
    labels = estimator.fit_predict(coords)
    labels_arr = np.asarray(labels)
    n_noise = int((labels_arr == -1).sum())
    unique_clusters = sorted({int(x) for x in labels_arr if x >= 0})
    cluster_sizes = {str(c): int((labels_arr == c).sum()) for c in unique_clusters}
    return {
        "method": "hdbscan",
        "available": True,
        "n_points": len(df),
        "min_cluster_size": min_cluster_size,
        "n_clusters": len(unique_clusters),
        "noise_fraction": float(n_noise / len(df)) if len(df) else 0.0,
        "cluster_sizes": cluster_sizes,
        "exposure_note": ("HDBSCAN clusters summarise recorded crash density, not road exposure."),
        "labels": labels_arr.tolist(),
    }


def bounded_kde_summary(
    df: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    """Optional Gaussian KDE on a bounded coordinate sample."""
    if len(df) == 0:
        return {"method": "kde", "skipped": True, "reason": "no_points"}
    try:
        from scipy.stats import gaussian_kde  # type: ignore[import-untyped]
    except ImportError:
        return {"method": "kde", "available": False}

    rng = np.random.default_rng(seed)
    n = min(sample_size, len(df))
    idx = rng.choice(len(df), size=n, replace=False)
    sample = df.iloc[idx]
    lat = pd.to_numeric(sample["crash_latitude"], errors="coerce").to_numpy()
    lon = pd.to_numeric(sample["crash_longitude"], errors="coerce").to_numpy()
    values = np.vstack([lat, lon])
    kde = gaussian_kde(values)
    densities = kde(values)
    return {
        "method": "kde",
        "sample_size": n,
        "density_min": float(np.min(densities)),
        "density_max": float(np.max(densities)),
        "density_median": float(np.median(densities)),
        "exposure_note": "KDE reflects recorded crash locations only.",
    }


def _dbscan_stability(df: pd.DataFrame, *, eps_meters: float, min_samples: int) -> dict[str, Any]:
    base = dbscan_cluster(df, eps_meters=eps_meters, min_samples=min_samples)
    if base.get("skipped"):
        return {"skipped": True}
    perturbed = dbscan_cluster(df, eps_meters=eps_meters * 1.1, min_samples=min_samples)
    labels_a = np.asarray(base.get("labels", []))
    labels_b = np.asarray(perturbed.get("labels", []))
    if len(labels_a) == 0 or len(labels_a) != len(labels_b):
        return {"skipped": True}
    ari = cluster_stability_ari(labels_a, labels_b)
    return {
        "eps_perturbation_pct": 10,
        "adjusted_rand_index": ari,
        "stable": ari >= 0.5,
    }


def run_hotspot_clustering(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run spatial clustering for configured methods and subsets."""
    started = time.perf_counter()
    metrics_path = artifacts_metrics_dir(paths) / f"hotspots_{config.profile}.json"

    if not force and metrics_path.is_file():
        logger.info("Hotspot outputs exist; skipping (use --force)")
        return {"status": "skipped", "metrics_path": str(metrics_path)}

    seed = config.seed
    set_random_seed(seed)
    methods = _configured_methods(config)
    tuning = config.tuning
    kde_sample = int(tuning.get("kde_sample_size", 5000))
    use_kde = bool(tuning.get("kde", False)) or config.profile == "extended"

    df = valid_coordinates(_load_spatial_frame(paths))
    logger.info("Hotspot analysis on %d coordinate-valid rows", len(df))

    subset_results: dict[str, Any] = {}
    for subset in SUBSET_DEFINITIONS:
        sub_df = filter_subset(df, subset)  # type: ignore[arg-type]
        if sub_df.empty and subset != "all":
            subset_results[subset] = {"skipped": True, "reason": "empty_subset", "n_points": 0}
            continue

        method_payload: dict[str, Any] = {
            "n_points": len(sub_df),
            "definition": SUBSET_DEFINITIONS[subset],
        }
        for method in methods:
            if method == "grid":
                method_payload["grid"] = grid_count_summary(sub_df)
            elif method == "dbscan":
                db = dbscan_cluster(sub_df)
                db.pop("labels", None)
                db["stability"] = _dbscan_stability(
                    sub_df,
                    eps_meters=DEFAULT_DBSCAN_EPS_METERS,
                    min_samples=DEFAULT_DBSCAN_MIN_SAMPLES,
                )
                method_payload["dbscan"] = db
            elif method == "hdbscan":
                hdb = hdbscan_cluster(sub_df)
                hdb.pop("labels", None)
                method_payload["hdbscan"] = hdb
            else:
                logger.warning("Unknown spatial method %s; skipping", method)

        if use_kde and len(sub_df) > 0:
            method_payload["kde"] = bounded_kde_summary(sub_df, sample_size=kde_sample, seed=seed)

        subset_results[subset] = method_payload

    elapsed = time.perf_counter() - started
    payload = {
        "task": "hotspots",
        "profile": config.profile,
        "timestamp_utc": utc_now_iso(),
        "seed": seed,
        "methods": methods,
        "subsets": subset_results,
        "exposure_disclaimer": (
            "All spatial outputs describe recorded-crash clusters only. "
            "Without traffic exposure they must not be read as intrinsic road-danger rankings."
        ),
        "timings": {"hotspot_clustering_seconds": elapsed},
    }
    persist_json(metrics_path, payload)
    logger.info("Hotspot clustering finished in %.2fs", elapsed)
    return {"status": "completed", **payload}
