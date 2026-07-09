"""Bounded model explanation: importance, PDP, stability, and ablations."""

from __future__ import annotations

import json
import time
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone  # type: ignore[import-untyped]
from sklearn.inspection import permutation_importance  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from crashlab.config import CrashlabConfig
from crashlab.data.manifest import utc_now_iso
from crashlab.evaluation.calibration import (
    apply_binary_threshold,
    brier_score,
    compare_calibration_methods,
)
from crashlab.evaluation.classification import _safe_pr_auc
from crashlab.evaluation.error_analysis import (
    build_fp_fn_tables,
    run_subgroup_analysis,
)
from crashlab.logging import get_logger
from crashlab.models.common import (
    artifacts_metrics_dir,
    load_moment_datasets,
    metrics_artifact_path,
    model_artifact_path,
    persist_json,
)
from crashlab.paths import CrashlabPaths

logger = get_logger("evaluation.explanation")

PREDICTIVE_NOTE = (
    "Feature effects and importances reflect predictive associations on held-out data, "
    "not causal mechanisms."
)

ABLATION_GROUPS: dict[str, list[str]] = {
    "no_location": [
        "loc_suburb",
        "loc_post_code",
        "loc_abs_statistical_area_2",
        "spatial_cell",
        "loc_latitude",
        "loc_longitude",
    ],
    "no_post_incident": [
        "crash_nature",
        "crash_type",
        "crash_dca_group_description",
        "dca_key_approach_dir",
        "count_unit_car",
        "count_unit_motorcycle_moped",
        "count_unit_truck",
        "count_unit_bus",
        "count_unit_bicycle",
        "count_unit_pedestrian",
        "count_unit_other",
    ],
    "no_weather_surface": [
        "crash_road_surface_condition",
        "crash_atmospheric_condition",
        "crash_lighting_condition",
    ],
    "no_high_cardinality": [
        "loc_suburb",
        "loc_abs_statistical_area_2",
        "spatial_cell",
    ],
}


def explanation_budget(config: CrashlabConfig) -> dict[str, Any]:
    """Resolved CPU bounds for explanation routines."""
    tuning = config.tuning
    seeds = config.project.get("random_seeds", [config.seed])
    if not isinstance(seeds, list):
        seeds = [config.seed]
    return {
        "permutation_repeats": int(tuning.get("permutation_repeats", 5)),
        "bootstrap_samples": int(tuning.get("bootstrap_samples", 100)),
        "pdp_max_features": int(tuning.get("pdp_max_features", 3)),
        "pdp_grid_points": int(tuning.get("pdp_grid_points", 15)),
        "skip_ablations": bool(tuning.get("skip_ablations", False)),
        "seed_stability_repeats": int(tuning.get("seed_stability_repeats", len(seeds))),
        "seeds": [int(s) for s in seeds[: max(1, int(tuning.get("seed_stability_repeats", 3)))]],
    }


def export_logistic_coefficients(
    model: Any,
    feature_names: list[str],
) -> dict[str, Any] | None:
    """Export standardized logistic coefficients when the estimator is linear."""
    estimator = model
    if isinstance(model, dict) and "model" in model:
        estimator = model["model"]
    if hasattr(estimator, "calibrated_classifiers_"):
        estimator = estimator.calibrated_classifiers_[0].estimator
    if not isinstance(estimator, LogisticRegression):
        return None
    coef = np.asarray(estimator.coef_).ravel()
    rows = [
        {
            "feature": feature_names[i] if i < len(feature_names) else f"f{i}",
            "coefficient": float(coef[i]),
        }
        for i in range(len(coef))
    ]
    rows.sort(key=lambda r: abs(float(str(r["coefficient"]))), reverse=True)
    return {
        "model_type": "logistic_regression",
        "top_coefficients": rows[:30],
        "interpretation_note": (
            "Coefficients describe associations in a linear probability model; "
            "they are not causal effects."
        ),
    }


def permutation_importance_held_out(
    model: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    *,
    n_repeats: int,
    seed: int,
    max_features: int = 15,
) -> dict[str, Any]:
    """Permutation importance on held-out test rows (bounded repeats)."""
    if len(y_test) < 10 or len(np.unique(y_test)) < 2:
        return {"skipped": True, "reason": "insufficient_test_rows"}
    n_repeats = max(1, min(n_repeats, 10))
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
        scoring="average_precision",
    )
    rows = [
        {
            "feature": feature_names[i] if i < len(feature_names) else f"f{i}",
            "importance_mean": float(result.importances_mean[i]),
            "importance_std": float(result.importances_std[i]),
        }
        for i in range(len(result.importances_mean))
    ]
    rows.sort(key=lambda r: float(str(r["importance_mean"])), reverse=True)
    return {
        "n_repeats": n_repeats,
        "top_features": rows[:max_features],
        "interpretation_note": PREDICTIVE_NOTE,
    }


def partial_dependence_1d(
    model: ClassifierMixin,
    X: np.ndarray,
    feature_index: int,
    feature_name: str,
    *,
    grid_points: int = 15,
) -> dict[str, Any]:
    """Single-feature partial dependence on a quantile grid."""
    if len(X) == 0:
        return {"feature": feature_name, "skipped": True}
    col = X[:, feature_index]
    grid = np.quantile(col, np.linspace(0.05, 0.95, grid_points))
    grid = np.unique(grid)
    X_base = X.copy()
    pd_values: list[float] = []
    for val in grid:
        X_mod = X_base.copy()
        X_mod[:, feature_index] = val
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_mod)[:, 1]
            pd_values.append(float(np.mean(proba)))
        else:
            pd_values.append(float(np.mean(model.predict(X_mod))))
    return {
        "feature": feature_name,
        "grid_values": [float(v) for v in grid],
        "partial_dependence": pd_values,
        "interpretation_note": PREDICTIVE_NOTE,
    }


def bootstrap_metric_cis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    ci: float = 0.9,
) -> dict[str, Any]:
    """Bootstrap confidence intervals for headline binary metrics."""
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true)
    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        proba = proba[:, 1]
    n = len(y_true_arr)
    if n < 20 or len(np.unique(y_true_arr)) < 2:
        return {"skipped": True, "reason": "insufficient_rows"}

    n_bootstrap = max(20, min(n_bootstrap, 500))
    pr_scores: list[float] = []
    brier_scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true_arr[idx]
        pp = proba[idx]
        pr = _safe_pr_auc(yt, pp)
        if pr is not None:
            pr_scores.append(pr)
        brier_scores.append(brier_score(yt, pp))

    alpha = (1.0 - ci) / 2

    def _ci(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"low": None, "median": None, "high": None}
        arr = np.asarray(values)
        return {
            "low": float(np.quantile(arr, alpha)),
            "median": float(np.median(arr)),
            "high": float(np.quantile(arr, 1 - alpha)),
        }

    return {
        "n_bootstrap": n_bootstrap,
        "ci_level": ci,
        "pr_auc": _ci(pr_scores),
        "brier": _ci(brier_scores),
        "note": "Intervals describe resampling uncertainty on the test split only.",
    }


def seed_stability_summary(
    build_estimator: Any,
    datasets: dict[str, Any],
    seeds: list[int],
) -> dict[str, Any]:
    """Refit a simple estimator across seeds and summarise validation PR-AUC spread."""
    train = datasets.get("train")
    val = datasets.get("val")
    if train is None or val is None:
        return {"skipped": True, "reason": "missing_splits"}
    scores: list[dict[str, Any]] = []
    for seed in seeds:
        est = clone(build_estimator)
        if hasattr(est, "random_state"):
            est.random_state = seed
        est.fit(train.X.to_numpy(), train.y)
        if not hasattr(est, "predict_proba"):
            continue
        pr = _safe_pr_auc(val.y, est.predict_proba(val.X.to_numpy()))
        scores.append({"seed": seed, "val_pr_auc": pr})
    if not scores:
        return {"skipped": True, "reason": "no_scores"}
    pr_values = [s["val_pr_auc"] for s in scores if s["val_pr_auc"] is not None]
    return {
        "seeds": seeds,
        "scores": scores,
        "val_pr_auc_range": {
            "min": float(min(pr_values)) if pr_values else None,
            "max": float(max(pr_values)) if pr_values else None,
            "std": float(np.std(pr_values)) if pr_values else None,
        },
        "note": "Seed stability reflects fitting variability, not deployment guarantees.",
    }


def _drop_feature_columns(X: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    keep = [c for c in X.columns if not any(c == p or c.startswith(f"{p}__") for p in prefixes)]
    return X[keep]


def run_feature_ablations(
    estimator: ClassifierMixin,
    datasets: dict[str, Any],
    *,
    groups: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Light ablation: drop one feature group and refit on training data."""
    groups = groups or ABLATION_GROUPS
    train = datasets.get("train")
    val = datasets.get("val")
    if train is None or val is None:
        return []
    baseline = clone(estimator)
    baseline.fit(train.X.to_numpy(), train.y)
    base_pr = (
        _safe_pr_auc(val.y, baseline.predict_proba(val.X.to_numpy()))
        if hasattr(baseline, "predict_proba")
        else None
    )
    rows: list[dict[str, Any]] = []
    for group_name, prefixes in groups.items():
        X_train = _drop_feature_columns(train.X, prefixes)
        X_val = _drop_feature_columns(val.X, prefixes)
        if X_train.shape[1] == 0:
            rows.append({"group": group_name, "skipped": True, "reason": "no_features_left"})
            continue
        model = clone(estimator)
        model.fit(X_train.to_numpy(), train.y)
        pr = (
            _safe_pr_auc(val.y, model.predict_proba(X_val.to_numpy()))
            if hasattr(model, "predict_proba")
            else None
        )
        delta = (pr - base_pr) if pr is not None and base_pr is not None else None
        rows.append(
            {
                "group": group_name,
                "dropped_prefixes": prefixes,
                "n_features_after": int(X_train.shape[1]),
                "val_pr_auc": pr,
                "delta_vs_full": delta,
                "interpretation_note": PREDICTIVE_NOTE,
            }
        )
    return rows


def _load_champion_binary(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    moment: str = "context",
) -> tuple[Any, dict[str, Any]] | None:
    metrics_path = metrics_artifact_path(paths, "binary", config.profile, moment)
    if not metrics_path.is_file():
        return None
    with metrics_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    champion = payload.get("champion")
    if not champion:
        return None
    model_name = champion.get("model_name")
    if not model_name:
        return None
    artifact_path = model_artifact_path(paths, "binary", moment, str(model_name))
    if not artifact_path.is_file():
        return None
    bundle = joblib.load(artifact_path)
    return bundle, champion


def run_explanation(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    force: bool = False,
    moment: str = "context",
) -> dict[str, Any]:
    """Run bounded explanation analysis for the binary champion model."""
    started = time.perf_counter()
    out_path = artifacts_metrics_dir(paths) / f"explanation_{config.profile}.json"
    if not force and out_path.is_file():
        logger.info("Explanation artifacts exist; skipping (use --force)")
        return {"status": "skipped", "path": str(out_path)}

    budget = explanation_budget(config)
    loaded = _load_champion_binary(config, paths, moment=moment)
    if loaded is None:
        payload = {
            "status": "no_champion",
            "profile": config.profile,
            "timestamp_utc": utc_now_iso(),
            "note": "Binary champion model not found; run train-binary first.",
        }
        persist_json(out_path, payload)
        return {"status": "completed", **payload}

    bundle, champion = loaded
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    threshold = float(bundle.get("threshold", 0.5)) if isinstance(bundle, dict) else 0.5

    datasets = load_moment_datasets(config, paths, moment, "severe_binary")
    test = datasets.get("test")
    train = datasets.get("train")
    val = datasets.get("val")

    result: dict[str, Any] = {
        "task": "explanation",
        "profile": config.profile,
        "timestamp_utc": utc_now_iso(),
        "moment": moment,
        "champion": {
            "model_name": champion.get("model_name"),
            "threshold": threshold,
        },
        "predictive_association_note": PREDICTIVE_NOTE,
        "budget": budget,
    }

    if test is not None and hasattr(model, "predict_proba"):
        X_test = test.X.to_numpy()
        y_test = test.y
        proba = model.predict_proba(X_test)
        pred = apply_binary_threshold(proba, threshold)
        feature_names = list(test.X.columns)

        result["permutation_importance"] = permutation_importance_held_out(
            model,
            X_test,
            y_test,
            feature_names,
            n_repeats=budget["permutation_repeats"],
            seed=config.seed,
        )

        imp = result["permutation_importance"].get("top_features", [])
        pdp_features = [f["feature"] for f in imp[: budget["pdp_max_features"]]]
        pdp_rows: list[dict[str, Any]] = []
        for fname in pdp_features:
            if fname in test.X.columns:
                idx = list(test.X.columns).index(fname)
                pdp_rows.append(
                    partial_dependence_1d(
                        model,
                        X_test,
                        idx,
                        fname,
                        grid_points=budget["pdp_grid_points"],
                    )
                )
        result["partial_dependence"] = pdp_rows

        result["bootstrap_cis"] = bootstrap_metric_cis(
            y_test,
            proba,
            n_bootstrap=budget["bootstrap_samples"],
            seed=config.seed,
        )

        meta_cols = [c for c in test.X.columns if c in test.X.columns]
        meta = test.X[meta_cols].copy()
        for col in ("crash_year", "loc_suburb", "crash_type", "crash_hour", "crash_severity"):
            if col not in meta.columns and col in datasets.get("test", test).X.columns:
                pass
        raw_test_rows = _align_metadata_frame(config, paths, moment, test)
        result["error_analysis"] = {
            "fp_fn_tables": build_fp_fn_tables(y_test, pred, proba, raw_test_rows),
            "subgroups": run_subgroup_analysis(
                y_test,
                pred,
                proba,
                raw_test_rows,
                years=test.years,
            ),
        }

        result["logistic_coefficients"] = export_logistic_coefficients(model, feature_names)

    if val is not None and train is not None and hasattr(model, "predict_proba"):
        base_est = champion.get("model_name")
        from crashlab.models.binary import build_binary_estimator

        est_template = build_binary_estimator(
            str(base_est),
            seed=config.seed,
            n_estimators_cap=int(config.tuning.get("n_estimators_cap", 100)),
        )
        result["calibration_comparison"] = compare_calibration_methods(
            est_template,
            train.X.to_numpy(),
            train.y,
            val.X.to_numpy(),
            val.y,
            test.X.to_numpy() if test else val.X.to_numpy(),
            test.y if test else val.y,
        )

    if not budget["skip_ablations"] and train is not None and val is not None:
        from crashlab.models.binary import build_binary_estimator

        est = build_binary_estimator(
            str(champion.get("model_name", "logistic")),
            seed=config.seed,
            n_estimators_cap=int(config.tuning.get("n_estimators_cap", 100)),
        )
        result["feature_ablations"] = run_feature_ablations(est, datasets)

    if budget["seed_stability_repeats"] > 0 and champion.get("model_name"):
        from crashlab.models.binary import build_binary_estimator

        def _builder() -> Any:
            return build_binary_estimator(
                str(champion["model_name"]),
                seed=config.seed,
                n_estimators_cap=int(config.tuning.get("n_estimators_cap", 100)),
            )

        result["seed_stability"] = seed_stability_summary(_builder(), datasets, budget["seeds"])

    elapsed = time.perf_counter() - started
    result["timings"] = {"explanation_seconds": elapsed}
    persist_json(out_path, result)
    logger.info("Explanation analysis finished in %.2fs", elapsed)
    return {"status": "completed", "path": str(out_path), **result}


def _align_metadata_frame(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    moment: str,
    test_split: Any,
) -> pd.DataFrame:
    """Attach human-readable columns for error review from processed parquet."""
    from crashlab.data.artifacts import processed_path
    from crashlab.features.build import prepare_derived_frame
    from crashlab.features.targets import add_binary_target, add_multiclass_target
    from crashlab.features.temporal import assign_split_column, compute_year_splits

    parquet = processed_path(paths)
    if not parquet.is_file():
        return pd.DataFrame(index=range(len(test_split.y)))
    df = pd.read_parquet(parquet)
    df = add_binary_target(add_multiclass_target(df))
    derived = prepare_derived_frame(df, moment)  # type: ignore[arg-type]
    years = sorted(int(y) for y in derived["crash_year"].dropna().unique())
    splits_cfg = config.raw.get("splits", {})
    if not isinstance(splits_cfg, dict):
        splits_cfg = {}
    train_end = splits_cfg.get("train_year_end")
    val_years_raw = splits_cfg.get("val_years")
    test_years_raw = splits_cfg.get("test_years")
    year_splits = compute_year_splits(
        years,
        train_year_end=int(train_end) if train_end is not None else None,
        val_years=[int(y) for y in val_years_raw] if isinstance(val_years_raw, list) else None,
        test_years=[int(y) for y in test_years_raw] if isinstance(test_years_raw, list) else None,
    )
    derived["split"] = assign_split_column(derived, year_splits)
    test_df = derived.loc[derived["split"] == "test"].reset_index(drop=True)
    if len(test_df) != len(test_split.y):
        return pd.DataFrame(index=range(len(test_split.y)))
    return test_df
