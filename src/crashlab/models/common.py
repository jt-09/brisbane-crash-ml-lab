"""Shared model training utilities, CV, champion selection, and persistence."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone  # type: ignore[import-untyped]
from sklearn.model_selection import (  # type: ignore[import-untyped]
    RandomizedSearchCV,
    StratifiedKFold,
)

from crashlab.config import CrashlabConfig
from crashlab.data.artifacts import processed_path
from crashlab.evaluation.calibration import (
    apply_binary_threshold,
    brier_score,
    calibration_curve_data,
    fit_calibrated_classifier,
)
from crashlab.evaluation.classification import _safe_pr_auc, recall_at_top_risk_pct
from crashlab.features.build import encoder_path, prepare_derived_frame
from crashlab.features.encoders import EncoderBundle, transform_with_mixed_encoding
from crashlab.features.temporal import YearSplits, assign_split_column, compute_year_splits
from crashlab.logging import get_logger
from crashlab.paths import CrashlabPaths

logger = get_logger("models.common")

LEADERBOARD_EXCLUDED_MOMENTS: frozenset[str] = frozenset({"leakage_demo"})
MAX_SEARCH_ITER = 15
MAX_SEARCH_FOLDS = 3

ModelFamily = Literal["binary", "multiclass", "ordinal"]


@dataclass(frozen=True)
class SplitDataset:
    """Feature matrix and labels for one temporal split."""

    X: pd.DataFrame
    y: np.ndarray
    years: np.ndarray


def set_random_seed(seed: int) -> None:
    """Set numpy RNG seed for reproducible training."""
    np.random.seed(seed)


def artifacts_metrics_dir(paths: CrashlabPaths) -> Path:
    directory = paths.artifacts_dir / "metrics"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def model_artifact_path(
    paths: CrashlabPaths,
    task: ModelFamily,
    moment: str,
    model_name: str,
) -> Path:
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    return paths.models_dir / f"{task}_{moment}_{model_name}.joblib"


def metrics_artifact_path(
    paths: CrashlabPaths,
    task: ModelFamily,
    profile: str,
    moment: str,
) -> Path:
    return artifacts_metrics_dir(paths) / f"{task}_{profile}_{moment}.json"


def tuning_params(config: CrashlabConfig) -> dict[str, int]:
    """Resolved hyperparameter search budget from profile config."""
    tuning = config.tuning
    n_iter = int(tuning.get("n_iter", 0))
    n_folds = int(tuning.get("n_folds", 0))
    cap = int(tuning.get("n_estimators_cap", 200))
    return {
        "n_iter": min(n_iter, MAX_SEARCH_ITER),
        "n_folds": min(n_folds, MAX_SEARCH_FOLDS) if n_folds > 0 else 0,
        "n_estimators_cap": cap,
    }


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


def load_moment_datasets(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    moment: str,
    target_col: str,
) -> dict[str, SplitDataset]:
    """Rebuild aligned feature matrices and labels for each temporal split."""
    parquet = processed_path(paths)
    if not parquet.is_file():
        msg = f"Processed parquet required: {parquet}"
        raise FileNotFoundError(msg)

    enc_path = encoder_path(paths, moment)
    if not enc_path.is_file():
        msg = f"Encoder bundle required for moment {moment}: {enc_path}"
        raise FileNotFoundError(msg)

    from crashlab.features.targets import add_binary_target, add_multiclass_target

    df = pd.read_parquet(parquet)
    df = add_binary_target(add_multiclass_target(df))
    derived = prepare_derived_frame(df, moment)  # type: ignore[arg-type]
    if target_col not in derived.columns:
        msg = f"Target column {target_col} missing from modeling frame"
        raise KeyError(msg)

    years = sorted(int(y) for y in derived["crash_year"].dropna().unique())
    year_splits = _resolve_year_splits(config, years)
    derived = derived.copy()
    derived["split"] = assign_split_column(derived, year_splits)

    bundle: EncoderBundle = joblib.load(enc_path)
    datasets: dict[str, SplitDataset] = {}
    for split_name in ("train", "val", "test"):
        split_df = derived.loc[derived["split"] == split_name]
        if split_df.empty:
            continue
        features = transform_with_mixed_encoding(bundle, split_df)
        target = split_df[target_col].to_numpy()
        valid = ~pd.isna(target)
        if not valid.all():
            features = features.loc[valid]
            target = target[valid]
            year_vals = split_df.loc[valid, "crash_year"].to_numpy()
        else:
            year_vals = split_df["crash_year"].to_numpy()
        datasets[split_name] = SplitDataset(
            X=features,
            y=target.astype(int),
            years=year_vals.astype(int),
        )
    return datasets


def build_train_cv(
    y_train: np.ndarray,
    *,
    n_folds: int,
    seed: int,
) -> StratifiedKFold | None:
    """Stratified K-fold on training rows only (years already held out)."""
    if n_folds < 2:
        return None
    n_splits = min(n_folds, MAX_SEARCH_FOLDS)
    unique, counts = np.unique(y_train, return_counts=True)
    min_class = int(counts.min()) if len(counts) else 0
    if min_class < n_splits:
        n_splits = max(2, min_class)
    if n_splits < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def fit_with_optional_search(
    estimator: BaseEstimator,
    param_distributions: dict[str, list[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_iter: int,
    cv: StratifiedKFold | None,
    seed: int,
    scoring: str = "average_precision",
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Fit estimator directly or via bounded RandomizedSearchCV."""
    if n_iter <= 0 or cv is None or not param_distributions:
        fitted = clone(estimator)
        start = time.perf_counter()
        fitted.fit(X_train, y_train)
        return fitted, {
            "search": "none",
            "fit_seconds": time.perf_counter() - start,
        }

    search = RandomizedSearchCV(
        estimator=clone(estimator),
        param_distributions=param_distributions,
        n_iter=min(n_iter, MAX_SEARCH_ITER),
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=seed,
        error_score="raise",
    )
    start = time.perf_counter()
    search.fit(X_train, y_train)
    return search.best_estimator_, {
        "search": "randomized",
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "fit_seconds": time.perf_counter() - start,
    }


def select_binary_threshold(
    y_val: np.ndarray,
    y_proba_val: np.ndarray,
) -> float:
    """Pick threshold maximising F1 on validation probabilities."""
    proba = np.asarray(y_proba_val)
    if proba.ndim == 2:
        proba = proba[:, 1]
    if len(np.unique(y_val)) < 2:
        return 0.5
    best_t = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = (proba >= threshold).astype(int)
        tp = ((pred == 1) & (y_val == 1)).sum()
        fp = ((pred == 1) & (y_val == 0)).sum()
        fn = ((pred == 0) & (y_val == 1)).sum()
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(threshold)
    return best_t


def _model_simplicity_score(model_name: str) -> int:
    order = {
        "dummy": 0,
        "logistic": 1,
        "multinomial_logistic": 1,
        "decision_tree": 2,
        "cumulative_logistic": 2,
        "random_forest": 3,
        "extra_trees": 4,
        "hist_gradient_boosting": 5,
        "cumulative_hgb": 5,
    }
    return order.get(model_name, 10)


def _champion_sort_key(candidate: dict[str, Any]) -> tuple:
    val = candidate.get("val_metrics", {})
    pr_auc = val.get("pr_auc")
    brier = val.get("brier")
    recall10 = val.get("recall_at_top_10pct")
    runtime = candidate.get("fit_seconds", 9999.0)
    return (
        -(pr_auc if pr_auc is not None else -1.0),
        brier if brier is not None else 999.0,
        -(recall10 if recall10 is not None else -1.0),
        _model_simplicity_score(str(candidate.get("model_name", ""))),
        runtime,
    )


def select_champion(
    candidates: list[dict[str, Any]],
    *,
    moment: str,
    baseline_pr_auc: float | None = None,
) -> dict[str, Any] | None:
    """Select champion using validation metric hierarchy; exclude leakage_demo."""
    if moment in LEADERBOARD_EXCLUDED_MOMENTS:
        return None

    eligible = [
        c
        for c in candidates
        if c.get("valid", True)
        and c.get("model_name") != "leakage_demo"
        and c.get("moment") != "leakage_demo"
    ]
    if not eligible:
        return None

    ranked = sorted(eligible, key=_champion_sort_key)
    champion = ranked[0]
    champion["is_champion"] = True
    if baseline_pr_auc is not None and champion.get("val_metrics", {}).get("pr_auc") is not None:
        champ_pr = float(champion["val_metrics"]["pr_auc"])
        champion["beats_dummy_pr_auc_relative"] = (
            (champ_pr - baseline_pr_auc) / baseline_pr_auc if baseline_pr_auc > 0 else None
        )
    return champion


def evaluate_binary_candidate(
    model: ClassifierMixin,
    datasets: dict[str, SplitDataset],
    *,
    threshold: float | None = None,
    calibrate: bool = True,
) -> tuple[ClassifierMixin, float, dict[str, Any]]:
    """Evaluate a fitted binary model; optionally calibrate on validation."""
    val = datasets.get("val")
    test = datasets.get("test")

    working = model
    chosen_threshold = threshold if threshold is not None else 0.5

    if calibrate and val is not None and hasattr(working, "predict_proba"):
        working = fit_calibrated_classifier(
            working,
            val.X.to_numpy(),
            val.y,
            method="isotonic",
        )

    if val is not None and threshold is None and hasattr(working, "predict_proba"):
        val_proba = working.predict_proba(val.X.to_numpy())
        chosen_threshold = select_binary_threshold(val.y, val_proba)

    metrics: dict[str, Any] = {"threshold": chosen_threshold}
    for split_name, split in (("val", val), ("test", test)):
        if split is None:
            continue
        if not hasattr(working, "predict_proba"):
            continue
        proba = working.predict_proba(split.X.to_numpy())
        pred = apply_binary_threshold(proba, chosen_threshold)
        from crashlab.evaluation.classification import binary_classification_metrics

        split_metrics = binary_classification_metrics(
            split.y,
            pred,
            proba,
            years=split.years,
        )
        split_metrics["brier"] = brier_score(split.y, proba)
        split_metrics["calibration_curve"] = calibration_curve_data(split.y, proba)
        metrics[split_name] = split_metrics

    return working, chosen_threshold, metrics


def persist_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def persist_model(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def dummy_baseline_pr_auc(candidates: list[dict[str, Any]]) -> float | None:
    """Extract dummy model validation PR-AUC for relative improvement checks."""
    for candidate in candidates:
        if candidate.get("model_name") == "dummy":
            pr = candidate.get("val_metrics", {}).get("pr_auc")
            if pr is not None:
                return float(pr)
    return None


def summarize_leaderboard(
    candidates: list[dict[str, Any]],
    *,
    moment: str,
) -> list[dict[str, Any]]:
    """Return sorted leaderboard rows excluding leakage_demo moment."""
    if moment in LEADERBOARD_EXCLUDED_MOMENTS:
        return []
    rows = []
    for candidate in candidates:
        if candidate.get("moment") == "leakage_demo":
            continue
        val = candidate.get("val_metrics", {})
        rows.append(
            {
                "model_name": candidate.get("model_name"),
                "moment": candidate.get("moment"),
                "val_pr_auc": val.get("pr_auc"),
                "val_brier": val.get("brier"),
                "val_recall_at_top_10pct": val.get("recall_at_top_10pct"),
                "is_champion": candidate.get("is_champion", False),
                "fit_seconds": candidate.get("fit_seconds"),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -(row["val_pr_auc"] if row["val_pr_auc"] is not None else -1.0),
            row["val_brier"] if row["val_brier"] is not None else 999.0,
        ),
    )


def quick_val_pr_auc(model: ClassifierMixin, X_val: np.ndarray, y_val: np.ndarray) -> float | None:
    """Validation PR-AUC for champion pre-selection."""
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(X_val)
    return _safe_pr_auc(y_val, proba)


def quick_val_recall_top10(
    model: ClassifierMixin,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(X_val)
    return recall_at_top_risk_pct(y_val, proba, 0.10)
