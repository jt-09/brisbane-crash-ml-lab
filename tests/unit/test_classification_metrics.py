"""Unit tests for classification metric helpers."""

from __future__ import annotations

import numpy as np

from crashlab.evaluation.calibration import (
    apply_binary_threshold,
    brier_score,
    calibration_curve_data,
)
from crashlab.evaluation.classification import (
    binary_classification_metrics,
    multiclass_classification_metrics,
    recall_at_top_risk_pct,
)
from crashlab.models.ordinal import cumulative_to_class_proba, enforce_monotone_cumulative


def test_recall_at_top_risk_pct() -> None:
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.4, 0.5, 0.6, 0.0])
    recall = recall_at_top_risk_pct(y_true, y_score, 0.20)
    assert recall is not None
    assert 0.0 <= recall <= 1.0


def test_binary_classification_metrics_shape() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0])
    proba = np.column_stack([1 - np.array([0.2, 0.3, 0.8, 0.1, 0.4]), [0.2, 0.3, 0.8, 0.1, 0.4]])
    metrics = binary_classification_metrics(y_true, y_pred, proba)
    assert metrics["n_samples"] == 5
    assert "pr_auc" in metrics
    assert len(metrics["confusion_matrix"]) == 2


def test_multiclass_metrics_include_kappa() -> None:
    y_true = np.array([0, 1, 2, 3, 1, 2])
    y_pred = np.array([0, 1, 2, 2, 1, 3])
    metrics = multiclass_classification_metrics(y_true, y_pred, labels=[0, 1, 2, 3])
    assert "macro_f1" in metrics
    assert "quadratic_weighted_kappa" in metrics
    assert "per_class" in metrics


def test_calibration_helpers() -> None:
    y_true = np.array([0, 1, 1, 0, 1, 0])
    proba = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3])
    curve = calibration_curve_data(y_true, proba, n_bins=3)
    assert "mean_predicted_value" in curve
    assert brier_score(y_true, proba) >= 0.0
    pred = apply_binary_threshold(proba, 0.5)
    assert set(pred.tolist()) <= {0, 1}


def test_ordinal_probability_monotonicity() -> None:
    cumulative = np.array([[0.9, 0.4, 0.6]])
    fixed = enforce_monotone_cumulative(cumulative)
    assert fixed[0, 1] <= fixed[0, 0]
    assert fixed[0, 2] <= fixed[0, 1]
    class_proba = cumulative_to_class_proba(cumulative)
    assert np.allclose(class_proba.sum(axis=1), 1.0)
