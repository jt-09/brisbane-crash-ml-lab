"""Classification metric helpers for binary, multiclass, and ordinal tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    cohen_kappa_score = None  # type: ignore[assignment,misc]


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true_arr = np.asarray(y_true)
    if len(np.unique(y_true_arr)) < 2:
        return None
    score = np.asarray(y_score)
    if score.ndim == 2 and score.shape[1] > 1:
        score = score[:, 1]
    return float(roc_auc_score(y_true_arr, score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true_arr = np.asarray(y_true)
    if len(np.unique(y_true_arr)) < 2:
        return None
    score = np.asarray(y_score)
    if score.ndim == 2 and score.shape[1] > 1:
        score = score[:, 1]
    return float(average_precision_score(y_true_arr, score))


def recall_at_top_risk_pct(
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_pct: float,
    *,
    positive_label: int = 1,
) -> float | None:
    """Fraction of positive labels captured in the top ``top_pct`` risk scores."""
    y_true_arr = np.asarray(y_true)
    score = np.asarray(y_score)
    if score.ndim == 2:
        score = score[:, positive_label]
    n_pos = int((y_true_arr == positive_label).sum())
    if n_pos == 0 or len(score) == 0:
        return None
    k = max(1, int(np.ceil(len(score) * top_pct)))
    order = np.argsort(-score)
    top = y_true_arr[order[:k]]
    return float((top == positive_label).sum() / n_pos)


def metrics_by_year(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    years: np.ndarray,
    *,
    y_score: np.ndarray | None = None,
) -> dict[str, dict[str, float | None]]:
    """Compute per-year binary metrics when year labels are available."""
    out: dict[str, dict[str, float | None]] = {}
    year_series = pd.Series(years)
    for year in sorted(year_series.dropna().unique()):
        mask = year_series == year
        if mask.sum() == 0:
            continue
        yt = y_true[mask.to_numpy()]
        yp = y_pred[mask.to_numpy()]
        entry: dict[str, float | None] = {
            "n": int(mask.sum()),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
        }
        if y_score is not None:
            ys = y_score[mask.to_numpy()]
            entry["pr_auc"] = _safe_pr_auc(yt, ys)
            entry["roc_auc"] = _safe_roc_auc(yt, ys)
        out[str(int(year))] = entry
    return out


def binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    years: np.ndarray | None = None,
    positive_label: int = 1,
) -> dict[str, Any]:
    """Full binary metric bundle for validation or test evaluation."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    proba = np.asarray(y_proba)
    pos_rate = float(np.mean(y_true_arr == positive_label)) if len(y_true_arr) else 0.0

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true_arr)),
        "positive_rate": pos_rate,
        "pr_auc": _safe_pr_auc(y_true_arr, proba),
        "roc_auc": _safe_roc_auc(y_true_arr, proba),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "recall_at_top_5pct": recall_at_top_risk_pct(y_true_arr, proba, 0.05),
        "recall_at_top_10pct": recall_at_top_risk_pct(y_true_arr, proba, 0.10),
        "recall_at_top_20pct": recall_at_top_risk_pct(y_true_arr, proba, 0.20),
    }
    if years is not None:
        metrics["by_year"] = metrics_by_year(y_true_arr, y_pred_arr, years, y_score=proba)
    return metrics


def multiclass_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list[int] | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Macro/weighted F1, per-class metrics, MAE, and quadratic weighted kappa."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    label_list = labels if labels is not None else sorted(int(v) for v in np.unique(y_true_arr))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=label_list,
        zero_division=0,
    )
    per_class: dict[str, dict[str, float | int]] = {}
    for idx, label in enumerate(label_list):
        name = class_names[idx] if class_names and idx < len(class_names) else str(label)
        per_class[name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    qwk: float | None = None
    if cohen_kappa_score is not None and len(y_true_arr):
        qwk = float(cohen_kappa_score(y_true_arr, y_pred_arr, weights="quadratic"))

    return {
        "n_samples": int(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "mean_absolute_class_error": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "quadratic_weighted_kappa": qwk,
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=label_list).tolist(),
        "per_class": per_class,
    }


def ordinal_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list[int] | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Ordinal metrics — same core set as multiclass with MAE emphasis."""
    base = multiclass_classification_metrics(
        y_true,
        y_pred,
        labels=labels,
        class_names=class_names,
    )
    base["task"] = "ordinal"
    return base
