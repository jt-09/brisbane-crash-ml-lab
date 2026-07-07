"""Ordinal severity classification with cumulative link models."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]

from crashlab.config import CrashlabConfig
from crashlab.data.manifest import utc_now_iso
from crashlab.evaluation.classification import ordinal_classification_metrics
from crashlab.features.targets import MULTICLASS_ORDER
from crashlab.logging import get_logger
from crashlab.models.common import (
    load_moment_datasets,
    metrics_artifact_path,
    model_artifact_path,
    persist_json,
    persist_model,
    set_random_seed,
    tuning_params,
)
from crashlab.paths import CrashlabPaths

logger = get_logger("models.ordinal")

ORDINAL_MOMENTS = ("context", "triage")
N_CLASSES = len(MULTICLASS_ORDER)
LABELS = list(range(N_CLASSES))


def ordinal_enabled(config: CrashlabConfig) -> bool:
    """Return whether ordinal training is enabled for the profile."""
    ordinal = config.models.get("ordinal")
    if ordinal is False:
        return False
    if isinstance(ordinal, list):
        return len(ordinal) > 0
    return bool(ordinal)


def _configured_models(config: CrashlabConfig) -> list[str]:
    ordinal = config.models.get("ordinal", [])
    if not isinstance(ordinal, list):
        return []
    return [str(m) for m in ordinal]


def enforce_monotone_cumulative(proba: np.ndarray) -> np.ndarray:
    """Ensure P(Y > k) is non-increasing across thresholds."""
    arr = np.asarray(proba, dtype=float).copy()
    if arr.ndim != 2:
        return arr
    for row_idx in range(arr.shape[0]):
        row = arr[row_idx]
        for idx in range(1, len(row)):
            row[idx] = min(row[idx], row[idx - 1])
        arr[row_idx] = row
    return arr


def cumulative_to_class_proba(cumulative: np.ndarray) -> np.ndarray:
    """Convert cumulative P(Y > k) to class probability mass."""
    cum = enforce_monotone_cumulative(cumulative)
    n_samples, n_thresholds = cum.shape
    class_proba = np.zeros((n_samples, n_thresholds + 1), dtype=float)
    class_proba[:, 0] = 1.0 - cum[:, 0]
    for k in range(1, n_thresholds):
        class_proba[:, k] = cum[:, k - 1] - cum[:, k]
    class_proba[:, -1] = cum[:, -1]
    class_proba = np.clip(class_proba, 0.0, 1.0)
    row_sums = class_proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return class_proba / row_sums


class CumulativeBinaryEnsemble(BaseEstimator, ClassifierMixin):
    """K-1 binary classifiers for cumulative ordinal probabilities."""

    def __init__(self, base_estimator: Any, n_classes: int = N_CLASSES, random_state: int = 42):
        self.base_estimator = base_estimator
        self.n_classes = n_classes
        self.random_state = random_state
        self.estimators_: list[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> CumulativeBinaryEnsemble:
        from sklearn.base import clone  # type: ignore[import-untyped]

        self.estimators_ = []
        y_arr = np.asarray(y, dtype=int)
        for threshold in range(self.n_classes - 1):
            binary_y = (y_arr > threshold).astype(int)
            est = clone(self.base_estimator)
            est.fit(X, binary_y)
            self.estimators_.append(est)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cumulative = np.column_stack([est.predict_proba(X)[:, 1] for est in self.estimators_])
        return cumulative_to_class_proba(cumulative)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class StatsmodelsOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Thin sklearn wrapper around statsmodels OrderedModel."""

    def __init__(self, n_classes: int = N_CLASSES):
        self.n_classes = n_classes
        self.result_: Any = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> StatsmodelsOrdinalClassifier:
        from statsmodels.miscmodels.ordinal_model import (  # type: ignore[import-untyped]
            OrderedModel,
        )

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        exog = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(X_arr.shape[1])])
        self.feature_names_ = list(exog.columns)
        model = OrderedModel(y_arr, exog, distr="logit")
        self.result_ = model.fit(method="bfgs", disp=False, maxiter=200)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.result_ is None:
            msg = "Model must be fitted before predict_proba"
            raise RuntimeError(msg)
        X_arr = np.asarray(X, dtype=float)
        exog = pd.DataFrame(X_arr, columns=self.feature_names_)
        return np.asarray(self.result_.predict(exog), dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def build_ordinal_estimator(model_name: str, *, seed: int, n_estimators_cap: int) -> Any:
    if model_name == "cumulative_logistic":
        return StatsmodelsOrdinalClassifier()
    if model_name == "cumulative_hgb":
        base = HistGradientBoostingClassifier(
            max_iter=min(100, n_estimators_cap),
            random_state=seed,
        )
        return CumulativeBinaryEnsemble(base, n_classes=N_CLASSES, random_state=seed)
    msg = f"Unknown ordinal model: {model_name}"
    raise ValueError(msg)


def _evaluate_ordinal(model: Any, datasets: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for split_name in ("val", "test"):
        split = datasets.get(split_name)
        if split is None:
            continue
        pred = model.predict(split.X.to_numpy())
        metrics[split_name] = ordinal_classification_metrics(
            split.y,
            pred,
            labels=LABELS,
            class_names=list(MULTICLASS_ORDER),
        )
    return metrics


def _select_ordinal_champion(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [c for c in candidates if c.get("valid", True)]
    if not eligible:
        return None
    ranked = sorted(
        eligible,
        key=lambda c: c.get("val_metrics", {}).get("mean_absolute_class_error") or 999.0,
    )
    champion = ranked[0]
    champion["is_champion"] = True
    return champion


def run_ordinal_training(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Train ordinal models when enabled in profile config."""
    if not ordinal_enabled(config):
        logger.info("Ordinal training disabled for profile %s", config.profile)
        return {"status": "skipped", "reason": "ordinal_disabled"}

    started = time.perf_counter()
    seed = config.seed
    set_random_seed(seed)
    budget = tuning_params(config)
    model_names = _configured_models(config)
    if not model_names:
        return {"status": "skipped", "reason": "no_ordinal_models"}

    results: dict[str, Any] = {
        "task": "ordinal",
        "profile": config.profile,
        "timestamp_utc": utc_now_iso(),
        "moments": {},
    }

    for moment in ORDINAL_MOMENTS:
        metrics_path = metrics_artifact_path(paths, "ordinal", config.profile, moment)
        if not force and metrics_path.is_file():
            logger.info("Ordinal metrics exist for %s; skipping (use --force)", moment)
            continue

        datasets = load_moment_datasets(config, paths, moment, "severity_class")
        if "train" not in datasets:
            msg = f"No training rows for ordinal moment {moment}"
            raise ValueError(msg)

        train = datasets["train"]
        candidates: list[dict[str, Any]] = []
        for model_name in model_names:
            try:
                start_fit = time.perf_counter()
                estimator = build_ordinal_estimator(
                    model_name,
                    seed=seed,
                    n_estimators_cap=budget["n_estimators_cap"],
                )
                estimator.fit(train.X.to_numpy(), train.y)
                fit_seconds = time.perf_counter() - start_fit
                eval_metrics = _evaluate_ordinal(estimator, datasets)
                model_path = model_artifact_path(paths, "ordinal", moment, model_name)
                persist_model(model_path, estimator)
                candidates.append(
                    {
                        "model_name": model_name,
                        "moment": moment,
                        "valid": True,
                        "fit_seconds": fit_seconds,
                        "val_metrics": eval_metrics.get("val", {}),
                        "test_metrics": eval_metrics.get("test", {}),
                        "model_path": str(model_path),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ordinal model %s/%s failed: %s", moment, model_name, exc)
                candidates.append(
                    {
                        "model_name": model_name,
                        "moment": moment,
                        "valid": False,
                        "error": str(exc),
                    }
                )

        champion = _select_ordinal_champion(candidates)
        moment_payload = {"candidates": candidates, "champion": champion}
        results["moments"][moment] = moment_payload
        persist_json(metrics_path, moment_payload)

    elapsed = time.perf_counter() - started
    results["timings"] = {"ordinal_training_seconds": elapsed}
    return {"status": "completed", **results}
