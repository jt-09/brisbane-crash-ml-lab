"""Multiclass severity classification training and evaluation."""

from __future__ import annotations

import time
from typing import Any

from sklearn.dummy import DummyClassifier  # type: ignore[import-untyped]
from sklearn.ensemble import (  # type: ignore[import-untyped]
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from crashlab.config import CrashlabConfig
from crashlab.data.manifest import utc_now_iso
from crashlab.evaluation.classification import multiclass_classification_metrics
from crashlab.features.targets import MULTICLASS_ORDER
from crashlab.logging import get_logger
from crashlab.models.common import (
    build_train_cv,
    fit_with_optional_search,
    load_moment_datasets,
    metrics_artifact_path,
    model_artifact_path,
    persist_json,
    persist_model,
    set_random_seed,
    tuning_params,
)
from crashlab.paths import CrashlabPaths

logger = get_logger("models.multiclass")

MULTICLASS_MOMENTS = ("context", "triage")
LABELS = list(range(len(MULTICLASS_ORDER)))


def _configured_models(config: CrashlabConfig) -> list[str]:
    models = config.models.get("multiclass", ["dummy"])
    if not isinstance(models, list):
        return ["dummy"]
    return [str(m) for m in models]


def build_multiclass_estimator(
    model_name: str,
    *,
    seed: int,
    n_estimators_cap: int,
) -> Any:
    if model_name == "dummy":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if model_name == "multinomial_logistic":
        return LogisticRegression(
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
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
            random_state=seed,
            max_iter=n_estimators_cap,
        )
    msg = f"Unknown multiclass model: {model_name}"
    raise ValueError(msg)


def _param_distributions(model_name: str, n_estimators_cap: int) -> dict[str, list[Any]]:
    if model_name == "multinomial_logistic":
        return {"C": [0.01, 0.1, 1.0, 10.0]}
    if model_name == "extra_trees":
        return {
            "n_estimators": [50, 100, min(150, n_estimators_cap)],
            "max_depth": [8, 12, 16],
        }
    if model_name == "hist_gradient_boosting":
        return {
            "max_iter": [50, 100, min(150, n_estimators_cap)],
            "learning_rate": [0.05, 0.1, 0.2],
        }
    return {}


def _evaluate_multiclass(
    model: Any,
    datasets: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for split_name in ("val", "test"):
        split = datasets.get(split_name)
        if split is None:
            continue
        pred = model.predict(split.X.to_numpy())
        metrics[split_name] = multiclass_classification_metrics(
            split.y,
            pred,
            labels=LABELS,
            class_names=list(MULTICLASS_ORDER),
        )
    return metrics


def _select_multiclass_champion(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [c for c in candidates if c.get("valid", True)]
    if not eligible:
        return None
    ranked = sorted(
        eligible,
        key=lambda c: (
            -(c.get("val_metrics", {}).get("macro_f1") or -1.0),
            c.get("val_metrics", {}).get("mean_absolute_class_error") or 999.0,
        ),
    )
    champion = ranked[0]
    champion["is_champion"] = True
    return champion


def run_multiclass_training(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Train configured multiclass models for context and triage moments."""
    started = time.perf_counter()
    seed = config.seed
    set_random_seed(seed)
    budget = tuning_params(config)
    model_names = _configured_models(config)

    results: dict[str, Any] = {
        "task": "multiclass",
        "profile": config.profile,
        "timestamp_utc": utc_now_iso(),
        "moments": {},
    }

    for moment in MULTICLASS_MOMENTS:
        metrics_path = metrics_artifact_path(paths, "multiclass", config.profile, moment)
        if not force and metrics_path.is_file():
            logger.info("Multiclass metrics exist for %s; skipping (use --force)", moment)
            continue

        datasets = load_moment_datasets(config, paths, moment, "severity_class")
        if "train" not in datasets:
            msg = f"No training rows for multiclass moment {moment}"
            raise ValueError(msg)

        train = datasets["train"]
        candidates: list[dict[str, Any]] = []
        for model_name in model_names:
            try:
                estimator = build_multiclass_estimator(
                    model_name,
                    seed=seed,
                    n_estimators_cap=budget["n_estimators_cap"],
                )
                cv = build_train_cv(train.y, n_folds=budget["n_folds"], seed=seed)
                fitted, fit_meta = fit_with_optional_search(
                    estimator,
                    _param_distributions(model_name, budget["n_estimators_cap"]),
                    train.X.to_numpy(),
                    train.y,
                    n_iter=budget["n_iter"],
                    cv=cv,
                    seed=seed,
                    scoring="f1_macro",
                )
                eval_metrics = _evaluate_multiclass(fitted, datasets)
                model_path = model_artifact_path(paths, "multiclass", moment, model_name)
                persist_model(model_path, fitted)
                candidates.append(
                    {
                        "model_name": model_name,
                        "moment": moment,
                        "valid": True,
                        "fit_seconds": fit_meta.get("fit_seconds"),
                        "search": fit_meta,
                        "val_metrics": eval_metrics.get("val", {}),
                        "test_metrics": eval_metrics.get("test", {}),
                        "model_path": str(model_path),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Multiclass model %s/%s failed: %s", moment, model_name, exc)
                candidates.append(
                    {
                        "model_name": model_name,
                        "moment": moment,
                        "valid": False,
                        "error": str(exc),
                    }
                )

        champion = _select_multiclass_champion(candidates)
        moment_payload = {"candidates": candidates, "champion": champion}
        results["moments"][moment] = moment_payload
        persist_json(metrics_path, moment_payload)

    elapsed = time.perf_counter() - started
    results["timings"] = {"multiclass_training_seconds": elapsed}
    return {"status": "completed", **results}
