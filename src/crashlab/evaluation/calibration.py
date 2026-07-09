"""Probability calibration helpers for classification models."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.base import ClassifierMixin  # type: ignore[import-untyped]
from sklearn.calibration import (  # type: ignore[import-untyped]
    CalibratedClassifierCV,
    calibration_curve,
)

CalibrationMethod = Literal["isotonic", "sigmoid"]


def fit_calibrated_classifier(
    estimator: ClassifierMixin,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    method: CalibrationMethod = "isotonic",
) -> CalibratedClassifierCV:
    """Fit a calibrated wrapper using validation data only."""
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv="prefit",
    )
    calibrated.fit(X_val, y_val)
    return calibrated


def calibration_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
    positive_label: int = 1,
) -> dict[str, Any]:
    """Return calibration curve arrays for persistence and plotting."""
    y_true_arr = np.asarray(y_true)
    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        proba = proba[:, positive_label]
    if len(np.unique(y_true_arr)) < 2:
        return {"fraction_of_positives": [], "mean_predicted_value": [], "n_bins": n_bins}
    fraction, mean_pred = calibration_curve(
        y_true_arr,
        proba,
        n_bins=n_bins,
        strategy="quantile",
    )
    return {
        "fraction_of_positives": [float(v) for v in fraction],
        "mean_predicted_value": [float(v) for v in mean_pred],
        "n_bins": n_bins,
    }


def brier_score(y_true: np.ndarray, y_proba: np.ndarray, *, positive_label: int = 1) -> float:
    """Binary Brier score from positive-class probabilities."""
    y_true_arr = np.asarray(y_true, dtype=float)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 2:
        proba = proba[:, positive_label]
    return float(np.mean((proba - y_true_arr) ** 2))


def apply_binary_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Classify using a decision threshold on positive-class probability."""
    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        proba = proba[:, 1]
    return (proba >= threshold).astype(int)


def compare_calibration_methods(
    estimator: ClassifierMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    methods: tuple[CalibrationMethod | None, ...] = (None, "sigmoid", "isotonic"),
) -> dict[str, Any]:
    """Compare uncalibrated vs sigmoid vs isotonic calibration on held-out data."""
    from sklearn.base import clone  # type: ignore[import-untyped]

    rows: list[dict[str, Any]] = []
    for method in methods:
        fitted = clone(estimator)
        fitted.fit(X_train, y_train)
        label = "none" if method is None else method
        if method is None:
            working: ClassifierMixin = fitted
        else:
            working = fit_calibrated_classifier(fitted, X_val, y_val, method=method)
        if not hasattr(working, "predict_proba"):
            rows.append({"method": label, "skipped": True})
            continue
        proba = working.predict_proba(X_eval)
        rows.append(
            {
                "method": label,
                "brier": brier_score(y_eval, proba),
                "calibration_curve": calibration_curve_data(y_eval, proba),
                "note": "Calibration adjusts probability scale; not a causal intervention.",
            }
        )
    best = min(
        (r for r in rows if r.get("brier") is not None),
        key=lambda r: float(r["brier"]),
        default=None,
    )
    return {
        "methods": rows,
        "best_method": best.get("method") if best else None,
        "interpretation_note": (
            "Lower Brier score indicates better probabilistic calibration on the evaluation split."
        ),
    }
