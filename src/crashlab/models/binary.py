"""Binary severity classification training and evaluation."""

from __future__ import annotations

import time
from typing import Any

from sklearn.dummy import DummyClassifier  # type: ignore[import-untyped]
from sklearn.ensemble import (  # type: ignore[import-untyped]
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import-untyped]

from crashlab.config import CrashlabConfig
from crashlab.data.manifest import utc_now_iso
from crashlab.evaluation.calibration import apply_binary_threshold, brier_score
from crashlab.evaluation.classification import binary_classification_metrics
from crashlab.logging import get_logger
from crashlab.models.common import (
    LEADERBOARD_EXCLUDED_MOMENTS,
    build_train_cv,
    dummy_baseline_pr_auc,
    evaluate_binary_candidate,
    fit_with_optional_search,
    load_moment_datasets,
    metrics_artifact_path,
    model_artifact_path,
    persist_json,
    persist_model,
    quick_val_pr_auc,
    quick_val_recall_top10,
    select_champion,
    set_random_seed,
    summarize_leaderboard,
    tuning_params,
)
from crashlab.paths import CrashlabPaths

logger = get_logger("models.binary")

BINARY_MOMENTS = ("context", "triage", "leakage_demo")


def _configured_models(config: CrashlabConfig) -> list[str]:
    models = config.models.get("binary", ["dummy", "logistic"])
    if not isinstance(models, list):
        return ["dummy"]
    return [str(m) for m in models]


def build_binary_estimator(
    model_name: str,
    *,
    seed: int,
    n_estimators_cap: int,
) -> Any:
    """Construct an unfitted binary classifier from config name."""
    if model_name == "dummy":
        return DummyClassifier(strategy="prior")
    if model_name == "logistic":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        )
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=8,
            random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced",
            n_estimators=n_estimators_cap,
            max_depth=12,
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            class_weight="balanced",
            n_estimators=n_estimators_cap,
            max_depth=12,
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=n_estimators_cap,
            random_state=seed,
        )
    msg = f"Unknown binary model: {model_name}"
    raise ValueError(msg)


def _param_distributions(model_name: str, n_estimators_cap: int) -> dict[str, list[Any]]:
    if model_name == "logistic":
        return {"C": [0.01, 0.1, 1.0, 10.0]}
    if model_name == "decision_tree":
        return {"max_depth": [4, 6, 8, 10], "min_samples_leaf": [5, 10, 20]}
    if model_name in {"random_forest", "extra_trees"}:
        return {
            "n_estimators": [50, 100, min(150, n_estimators_cap)],
            "max_depth": [8, 12, 16],
            "min_samples_leaf": [1, 5, 10],
        }
    if model_name == "hist_gradient_boosting":
        return {
            "max_iter": [50, 100, min(150, n_estimators_cap)],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        }
    return {}


def _train_one_binary_model(
    model_name: str,
    moment: str,
    datasets: dict[str, Any],
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    seed: int,
    budget: dict[str, int],
) -> dict[str, Any]:
    train = datasets["train"]
    val = datasets.get("val")
    X_train = train.X.to_numpy()
    y_train = train.y

    estimator = build_binary_estimator(
        model_name,
        seed=seed,
        n_estimators_cap=budget["n_estimators_cap"],
    )
    cv = build_train_cv(y_train, n_folds=budget["n_folds"], seed=seed)
    fitted, fit_meta = fit_with_optional_search(
        estimator,
        _param_distributions(model_name, budget["n_estimators_cap"]),
        X_train,
        y_train,
        n_iter=budget["n_iter"],
        cv=cv,
        seed=seed,
    )

    val_metrics: dict[str, Any] = {}
    if val is not None and hasattr(fitted, "predict_proba"):
        val_proba = fitted.predict_proba(val.X.to_numpy())
        val_pred = apply_binary_threshold(val_proba, 0.5)
        val_metrics = binary_classification_metrics(val.y, val_pred, val_proba, years=val.years)
        val_metrics["brier"] = brier_score(val.y, val_proba)
        val_metrics["pr_auc"] = quick_val_pr_auc(fitted, val.X.to_numpy(), val.y)
        val_metrics["recall_at_top_10pct"] = quick_val_recall_top10(fitted, val.X.to_numpy(), val.y)

    calibrated, threshold, eval_metrics = evaluate_binary_candidate(
        fitted,
        datasets,
        calibrate=True,
    )

    model_path = model_artifact_path(paths, config.profile, "binary", moment, model_name)
    persist_model(model_path, {"model": calibrated, "threshold": threshold, "moment": moment})

    candidate = {
        "model_name": model_name,
        "moment": moment,
        "valid": True,
        "fit_seconds": fit_meta.get("fit_seconds"),
        "search": fit_meta,
        "threshold": threshold,
        "val_metrics": val_metrics or eval_metrics.get("val", {}),
        "test_metrics": eval_metrics.get("test", {}),
        "model_path": str(model_path),
    }
    return candidate


def run_binary_training(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Train configured binary models for each prediction moment."""
    started = time.perf_counter()
    seed = config.seed
    set_random_seed(seed)
    budget = tuning_params(config)
    model_names = _configured_models(config)

    results: dict[str, Any] = {
        "task": "binary",
        "profile": config.profile,
        "timestamp_utc": utc_now_iso(),
        "moments": {},
    }

    for moment in BINARY_MOMENTS:
        metrics_path = metrics_artifact_path(paths, "binary", config.profile, moment)
        if not force and metrics_path.is_file() and moment not in LEADERBOARD_EXCLUDED_MOMENTS:
            logger.info("Binary metrics exist for %s; skipping (use --force)", moment)
            continue

        logger.info("Training binary models for moment=%s models=%s", moment, model_names)
        datasets = load_moment_datasets(config, paths, moment, "severe_binary")
        if "train" not in datasets:
            msg = f"No training rows for binary moment {moment}"
            raise ValueError(msg)

        candidates: list[dict[str, Any]] = []
        for model_name in model_names:
            try:
                candidate = _train_one_binary_model(
                    model_name,
                    moment,
                    datasets,
                    config,
                    paths,
                    seed=seed,
                    budget=budget,
                )
                candidates.append(candidate)
                logger.info(
                    "  %s/%s val PR-AUC=%s",
                    moment,
                    model_name,
                    candidate.get("val_metrics", {}).get("pr_auc"),
                )
            except Exception as exc:  # noqa: BLE001 — collect failure per model
                logger.warning("Binary model %s/%s failed: %s", moment, model_name, exc)
                candidates.append(
                    {
                        "model_name": model_name,
                        "moment": moment,
                        "valid": False,
                        "error": str(exc),
                    }
                )

        baseline = dummy_baseline_pr_auc(candidates)
        champion = select_champion(candidates, moment=moment, baseline_pr_auc=baseline)
        leaderboard = summarize_leaderboard(candidates, moment=moment)

        moment_payload = {
            "candidates": candidates,
            "champion": champion,
            "leaderboard": leaderboard,
            "baseline_pr_auc": baseline,
            "excluded_from_leaderboard": moment in LEADERBOARD_EXCLUDED_MOMENTS,
        }
        results["moments"][moment] = moment_payload
        persist_json(metrics_path, moment_payload)

    elapsed = time.perf_counter() - started
    results["timings"] = {"binary_training_seconds": elapsed}
    logger.info("Binary training finished in %.2fs", elapsed)
    return {"status": "completed", **results}
