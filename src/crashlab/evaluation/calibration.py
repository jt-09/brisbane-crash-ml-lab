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
