"""False-positive / false-negative review and subgroup performance tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crashlab.evaluation.classification import binary_classification_metrics

MIN_SUBGROUP_N = 30
PREDICTIVE_NOTE = (
    "Subgroup metrics describe predictive associations on held-out data only; "
    "they are not causal effects and may be unreliable for small groups."
)


def build_fp_fn_tables(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metadata: pd.DataFrame,
    *,
    max_rows: int = 25,
    positive_label: int = 1,
) -> dict[str, Any]:
    """Build capped FP/FN review tables with optional metadata columns."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        proba = proba[:, positive_label]

    fp_mask = (y_pred_arr == positive_label) & (y_true_arr != positive_label)
    fn_mask = (y_pred_arr != positive_label) & (y_true_arr == positive_label)

    meta_cols = [
        c
        for c in (
            "crash_year",
            "crash_month",
            "crash_day_of_week",
            "loc_suburb",
            "crash_type",
            "crash_severity",
        )
        if c in metadata.columns
    ]

    def _table(mask: np.ndarray, kind: str) -> list[dict[str, Any]]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        order = idx[np.argsort(-proba[idx])] if kind == "fp" else idx[np.argsort(proba[idx])]
        order = order[:max_rows]
        rows: list[dict[str, Any]] = []
        for i in order:
            row: dict[str, Any] = {
                "row_index": int(i),
                "predicted_proba": float(proba[i]),
                "true_label": int(y_true_arr[i]),
                "predicted_label": int(y_pred_arr[i]),
            }
            for col in meta_cols:
                val = metadata.iloc[i][col]
                if hasattr(val, "item"):
                    val = val.item()
                row[col] = val
            rows.append(row)
        return rows

    return {
        "false_positives": _table(fp_mask, "fp"),
        "false_negatives": _table(fn_mask, "fn"),
        "n_false_positives": int(fp_mask.sum()),
        "n_false_negatives": int(fn_mask.sum()),
        "max_rows_shown": max_rows,
        "interpretation_note": PREDICTIVE_NOTE,
    }


def subgroup_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    subgroup: pd.Series,
    *,
    min_n: int = MIN_SUBGROUP_N,
    years: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Per-subgroup binary metrics with reliability flags for small n."""
    rows: list[dict[str, Any]] = []
    subgroup_arr = subgroup.reset_index(drop=True)
    for value in subgroup_arr.unique():
        mask_arr = (subgroup_arr == value).to_numpy()
        n = int(mask_arr.sum())
        if n == 0:
            continue
        yt = y_true[mask_arr]
        yp = y_pred[mask_arr]
        prob = y_proba[mask_arr]
        year_vals = years[mask_arr] if years is not None else None
        metrics = binary_classification_metrics(yt, yp, prob, years=year_vals)
        rows.append(
            {
                "subgroup": str(value),
                "n": n,
                "reliable": n >= min_n,
                "pr_auc": metrics.get("pr_auc"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "f1": metrics.get("f1"),
                "positive_rate": metrics.get("positive_rate"),
                "caveat": None
                if n >= min_n
                else f"Small n={n}; metrics may be unstable (min_n={min_n}).",
            }
        )
    return sorted(rows, key=lambda r: (-int(r["reliable"]), -r["n"]))


def run_subgroup_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    frame: pd.DataFrame,
    *,
    min_n: int = MIN_SUBGROUP_N,
    years: np.ndarray | None = None,
) -> dict[str, Any]:
    """Subgroup slices: year, broad time period, suburb (when large enough)."""
    out: dict[str, Any] = {"note": PREDICTIVE_NOTE, "slices": {}}

    if "crash_year" in frame.columns:
        out["slices"]["by_year"] = subgroup_performance(
            y_true,
            y_pred,
            y_proba,
            frame["crash_year"].astype(str),
            min_n=min_n,
            years=years,
        )

    if "crash_hour" in frame.columns:
        hour = pd.to_numeric(frame["crash_hour"], errors="coerce")

        def _period(h: float) -> str:
            if pd.isna(h):
                return "unknown"
            h_int = int(h)
            if 6 <= h_int < 12:
                return "morning"
            if 12 <= h_int < 18:
                return "afternoon"
            if 18 <= h_int < 22:
                return "evening"
            return "night_early"

        out["slices"]["by_time_period"] = subgroup_performance(
            y_true,
            y_pred,
            y_proba,
            hour.map(_period),
            min_n=min_n,
            years=years,
        )

    if "loc_suburb" in frame.columns:
        suburb_counts = frame["loc_suburb"].value_counts()
        large_suburbs = set(suburb_counts[suburb_counts >= min_n].index.astype(str))
        suburb_series = (
            frame["loc_suburb"]
            .astype(str)
            .where(
                frame["loc_suburb"].astype(str).isin(large_suburbs),
                other="(other_small_suburbs)",
            )
        )
        out["slices"]["by_suburb_large_only"] = subgroup_performance(
            y_true,
            y_pred,
            y_proba,
            suburb_series,
            min_n=min_n,
            years=years,
        )

    if "crash_type" in frame.columns:
        out["slices"]["by_crash_type"] = subgroup_performance(
            y_true,
            y_pred,
            y_proba,
            frame["crash_type"].fillna("(missing)").astype(str),
            min_n=min_n,
            years=years,
        )

    return out
