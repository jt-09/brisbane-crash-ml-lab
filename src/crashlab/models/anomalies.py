"""Unsupervised anomaly detection on cleaned crash records."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]
from sklearn.neighbors import LocalOutlierFactor  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from crashlab.config import CrashlabConfig
from crashlab.data.artifacts import processed_path
from crashlab.data.manifest import utc_now_iso
from crashlab.data.schema import HOUR_MAX, HOUR_MIN, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN
from crashlab.evaluation.stability import top_k_jaccard
from crashlab.features.targets import add_binary_target, filter_modeling_rows
from crashlab.features.temporal import add_cyclic_time_features
from crashlab.logging import get_logger
from crashlab.models.common import artifacts_metrics_dir, persist_json, set_random_seed
from crashlab.paths import CrashlabPaths

logger = get_logger("models.anomalies")

REVIEW_TOP_N = 50
STABILITY_TOP_K = 100
STABILITY_SEEDS = (42, 7)

RULE_FEATURES = (
    "crash_hour",
    "crash_longitude",
    "crash_latitude",
    "crash_speed_limit",
    "crash_year",
    "crash_month",
)


def _configured_methods(config: CrashlabConfig) -> list[str]:
    methods = config.models.get("anomalies", ["rules"])
    if not isinstance(methods, list):
        return ["rules"]
    return [str(m) for m in methods]


def _load_modeling_frame(paths: CrashlabPaths, profile: str) -> pd.DataFrame:
    parquet = processed_path(paths, profile)
    if not parquet.is_file():
        msg = f"Processed parquet required: {parquet}"
        raise FileNotFoundError(msg)
    df = pd.read_parquet(parquet)
    df = filter_modeling_rows(df)
    df = add_binary_target(df)
    df = add_cyclic_time_features(df)
    return df.reset_index(drop=True)


def _numeric_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in RULE_FEATURES if c in df.columns]
    matrix = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return matrix, cols


def _series_column(df: pd.DataFrame, column: str, default: float | None = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _robust_z_scores(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    median = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - median))
    if mad < 1e-9:
        return np.zeros_like(arr)
    return np.asarray(0.6745 * (arr - median) / mad, dtype=float)


def rule_anomaly_scores(df: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
    """Score records using validation-style and rarity rules (higher = more anomalous)."""
    n = len(df)
    scores = np.zeros(n, dtype=float)
    reasons: list[list[str]] = [[] for _ in range(n)]

    def bump(idx: int, weight: float, reason: str) -> None:
        scores[idx] += weight
        reasons[idx].append(reason)

    hour = _series_column(df, "crash_hour")
    lon = _series_column(df, "crash_longitude", default=np.nan)
    lat = _series_column(df, "crash_latitude", default=np.nan)
    speed = _series_column(df, "crash_speed_limit", default=np.nan)

    for i in range(n):
        h = hour.iloc[i] if i < len(hour) else np.nan
        if pd.notna(h) and (h <= HOUR_MIN + 1 or h >= HOUR_MAX - 1):
            bump(i, 1.0, "hour_near_range_edge")
        lo = lon.iloc[i] if i < len(lon) else np.nan
        la = lat.iloc[i] if i < len(lat) else np.nan
        if pd.notna(lo) and (lo <= LON_MIN + 0.02 or lo >= LON_MAX - 0.02):
            bump(i, 1.5, "longitude_near_bbox_edge")
        if pd.notna(la) and (la <= LAT_MIN + 0.02 or la >= LAT_MAX - 0.02):
            bump(i, 1.5, "latitude_near_bbox_edge")
        sp = speed.iloc[i] if i < len(speed) else np.nan
        if pd.notna(sp) and sp >= 100:
            bump(i, 1.0, "high_speed_limit")

    matrix, cols = _numeric_matrix(df)
    for col in cols:
        z = np.abs(_robust_z_scores(matrix[col].to_numpy()))
        for i, zval in enumerate(z):
            if zval >= 3.5:
                bump(i, min(zval / 3.5, 3.0), f"robust_z_{col}")

    if "loc_suburb" in df.columns and "crash_hour" in df.columns:
        combo = df["loc_suburb"].astype(str) + "|" + hour.astype(str)
        freq = combo.value_counts()
        rare = combo.map(freq) <= 1
        for i, is_rare in enumerate(rare):
            if bool(is_rare):
                bump(i, 0.5, "rare_suburb_hour_combo")

    return scores, reasons


def robust_z_anomaly_scores(df: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
    """Aggregate robust univariate z-scores across numeric fields."""
    matrix, cols = _numeric_matrix(df)
    n = len(df)
    scores = np.zeros(n, dtype=float)
    reasons: list[list[str]] = [[] for _ in range(n)]
    for col in cols:
        z = np.abs(_robust_z_scores(matrix[col].to_numpy()))
        for i, zval in enumerate(z):
            if zval > scores[i]:
                scores[i] = zval
                reasons[i] = [f"max_robust_z_{col}={zval:.2f}"]
            elif zval >= scores[i] * 0.95 and zval > 2.0:
                reasons[i].append(f"robust_z_{col}={zval:.2f}")
    return scores, reasons


def _scaled_features(df: pd.DataFrame) -> np.ndarray:
    matrix, _ = _numeric_matrix(df)
    scaler = StandardScaler()
    return np.asarray(scaler.fit_transform(matrix.to_numpy()), dtype=float)


def isolation_forest_scores(
    df: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int = 100,
) -> np.ndarray:
    """Isolation Forest anomaly scores (higher = more anomalous)."""
    x = _scaled_features(df)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x)
    raw = model.score_samples(x)
    return np.asarray(-raw, dtype=float)


def lof_scores(
    df: pd.DataFrame,
    *,
    seed: int,
    sample_size: int,
    n_neighbors: int = 20,
) -> np.ndarray:
    """LOF scores on a bounded sample; non-sample rows receive neutral score."""
    n = len(df)
    scores = np.zeros(n, dtype=float)
    if n == 0:
        return scores
    rng = np.random.default_rng(seed)
    size = min(sample_size, n)
    sample_idx = rng.choice(n, size=size, replace=False)
    x = _scaled_features(df)
    x_sample = x[sample_idx]
    neighbors = min(n_neighbors, max(2, size - 1))
    model = LocalOutlierFactor(
        n_neighbors=neighbors,
        contamination="auto",
        novelty=False,
        n_jobs=-1,
    )
    model.fit_predict(x_sample)
    sample_scores = -model.negative_outlier_factor_
    scores[sample_idx] = sample_scores
    return scores


def _build_review_table(
    df: pd.DataFrame,
    method: str,
    scores: np.ndarray,
    reason_lists: list[list[str]] | None,
    *,
    top_n: int = REVIEW_TOP_N,
) -> list[dict[str, Any]]:
    order = np.argsort(-scores, kind="stable")[:top_n]
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(order, start=1):
        row = df.iloc[int(idx)]
        reasons = reason_lists[int(idx)] if reason_lists else []
        rows.append(
            {
                "rank": rank,
                "method": method,
                "crash_ref_number": row.get("crash_ref_number"),
                "anomaly_score": float(scores[int(idx)]),
                "crash_year": row.get("crash_year"),
                "crash_month": row.get("crash_month"),
                "crash_hour": row.get("crash_hour"),
                "loc_suburb": row.get("loc_suburb"),
                "crash_severity": row.get("crash_severity"),
                "reason_features": "; ".join(reasons) if reasons else "model_score",
                "review_category": "unassigned",
            }
        )
    return rows


def _isolation_forest_stability(
    df: pd.DataFrame,
    *,
    n_estimators: int,
    top_k: int = STABILITY_TOP_K,
) -> dict[str, Any]:
    scores_by_seed: dict[int, np.ndarray] = {}
    for seed in STABILITY_SEEDS:
        scores_by_seed[seed] = isolation_forest_scores(df, seed=seed, n_estimators=n_estimators)
    jaccard = top_k_jaccard(
        scores_by_seed[STABILITY_SEEDS[0]], scores_by_seed[STABILITY_SEEDS[1]], top_k
    )
    stable = jaccard >= 0.25
    return {
        "seeds": list(STABILITY_SEEDS),
        "top_k": top_k,
        "jaccard": float(jaccard),
        "stable": stable,
        "note": None if stable else "Top-k overlap below 0.25 — interpret rankings cautiously.",
    }


def run_anomaly_detection(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run configured anomaly detectors and write review tables."""
    started = time.perf_counter()
    metrics_path = artifacts_metrics_dir(paths) / f"anomalies_{config.profile}.json"
    review_path = paths.tables_dir / f"anomaly_review_{config.profile}.csv"

    if not force and metrics_path.is_file() and review_path.is_file():
        logger.info("Anomaly outputs exist; skipping (use --force)")
        return {"status": "skipped", "metrics_path": str(metrics_path)}

    seed = config.seed
    set_random_seed(seed)
    methods = _configured_methods(config)
    tuning = config.tuning
    lof_sample = int(tuning.get("lof_sample_size", 5000))
    n_estimators = int(tuning.get("n_estimators_cap", 100))

    df = _load_modeling_frame(paths, config.profile)
    logger.info("Anomaly detection on %d rows, methods=%s", len(df), methods)

    method_results: dict[str, Any] = {}
    all_review_rows: list[dict[str, Any]] = []

    for method in methods:
        reason_lists: list[list[str]] | None = None
        if method == "rules":
            scores, reason_lists = rule_anomaly_scores(df)
        elif method == "robust_z":
            scores, reason_lists = robust_z_anomaly_scores(df)
        elif method == "isolation_forest":
            scores = isolation_forest_scores(df, seed=seed, n_estimators=n_estimators)
        elif method == "lof":
            scores = lof_scores(df, seed=seed, sample_size=lof_sample)
            reason_lists = [
                ["lof_sample_member"] if s > 0 else ["not_in_lof_sample"] for s in scores
            ]
        else:
            logger.warning("Unknown anomaly method %s; skipping", method)
            continue

        review = _build_review_table(df, method, scores, reason_lists)
        all_review_rows.extend(review)
        entry: dict[str, Any] = {
            "method": method,
            "n_rows": len(df),
            "score_min": float(np.min(scores)) if len(scores) else 0.0,
            "score_max": float(np.max(scores)) if len(scores) else 0.0,
            "top_score_mean": float(np.mean(scores[top_k_indices_local(scores, REVIEW_TOP_N)]))
            if len(scores)
            else 0.0,
            "review_top_n": REVIEW_TOP_N,
        }
        if method == "isolation_forest":
            entry["stability"] = _isolation_forest_stability(
                df, n_estimators=n_estimators, top_k=STABILITY_TOP_K
            )
        method_results[method] = entry

    review_df = pd.DataFrame(all_review_rows)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(review_path, index=False)

    elapsed = time.perf_counter() - started
    payload = {
        "task": "anomalies",
        "profile": config.profile,
        "timestamp_utc": utc_now_iso(),
        "seed": seed,
        "methods": methods,
        "n_rows": len(df),
        "methods_detail": method_results,
        "review_table_path": str(review_path),
        "timings": {"anomaly_detection_seconds": elapsed},
    }
    persist_json(metrics_path, payload)
    logger.info("Anomaly detection finished in %.2fs", elapsed)
    return {"status": "completed", **payload}


def top_k_indices_local(scores: np.ndarray, k: int) -> np.ndarray:
    """Local top-k helper to avoid circular imports in summaries."""
    if k <= 0 or len(scores) == 0:
        return np.array([], dtype=int)
    k = min(k, len(scores))
    return np.argsort(-scores, kind="stable")[:k]
