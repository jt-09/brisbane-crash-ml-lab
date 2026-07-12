"""Unit tests for classification metric helpers."""

from __future__ import annotations

import numpy as np

from crashlab.evaluation.calibration import (
    apply_binary_threshold,
    brier_score,
    calibration_curve_data,
)
from crashlab.evaluation.classification import (
    _safe_pr_auc,
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


def test_safe_pr_auc_constant_scores_match_prevalence() -> None:
    """Constant-score baselines must not inflate PR-AUC above prevalence."""
    from sklearn.dummy import DummyClassifier

    rng = np.random.default_rng(0)
    y_true = rng.choice([0, 1], size=500, p=[0.6, 0.4])
    X = rng.normal(size=(500, 3))
    dummy = DummyClassifier(strategy="prior").fit(X, y_true)
    proba = dummy.predict_proba(X)
    pr_auc = _safe_pr_auc(y_true, proba)
    assert pr_auc is not None
    assert abs(pr_auc - float(y_true.mean())) < 1e-6


def test_safe_pr_auc_ranked_scores_exceed_prevalence() -> None:
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45])
    pr_auc = _safe_pr_auc(y_true, y_score)
    assert pr_auc is not None
    assert pr_auc > 0.4


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
