"""Suburb-month crash count models with strictly historical lags.

Predictions concern *recorded* crash counts aggregated by suburb and month.
Without traffic exposure, results must not be interpreted as intrinsic road risk.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    mean_absolute_error,
    mean_poisson_deviance,
)

from crashlab.config import CrashlabConfig
from crashlab.data.artifacts import processed_path
from crashlab.data.manifest import utc_now_iso
from crashlab.features.targets import add_binary_target, filter_modeling_rows
from crashlab.features.temporal import YearSplits, assign_split_column, compute_year_splits
from crashlab.logging import get_logger
from crashlab.models.common import artifacts_metrics_dir, persist_json, set_random_seed
from crashlab.paths import CrashlabPaths

logger = get_logger("models.counts")

CountTarget = Literal["crash_count", "severe_count"]


def _configured_models(config: CrashlabConfig) -> list[str]:
    models = config.models.get("counts", ["seasonal_mean"])
    if not isinstance(models, list):
        return ["seasonal_mean"]
    return [str(m) for m in models]


def _load_frame(paths: CrashlabPaths, profile: str) -> pd.DataFrame:
    parquet = processed_path(paths, profile)
    if not parquet.is_file():
        msg = f"Processed parquet required: {parquet}"
        raise FileNotFoundError(msg)
    df = pd.read_parquet(parquet)
    return add_binary_target(filter_modeling_rows(df))


def build_count_panel(df: pd.DataFrame, *, target: CountTarget = "crash_count") -> pd.DataFrame:
    """Aggregate suburb × year × month counts with period ordering."""
    if "loc_suburb" not in df.columns:
        msg = "loc_suburb required for count aggregation"
        raise KeyError(msg)
    grouped = (
        df.groupby(["loc_suburb", "crash_year", "crash_month"], dropna=False)
        .agg(
            crash_count=("crash_ref_number", "count"),
            severe_count=("severe_binary", "sum"),
        )
        .reset_index()
    )
    grouped["crash_year"] = grouped["crash_year"].astype(int)
    grouped["crash_month"] = grouped["crash_month"].astype(int)
    grouped["period"] = grouped["crash_year"] * 12 + grouped["crash_month"]
    grouped = grouped.sort_values(["loc_suburb", "period"]).reset_index(drop=True)
    if target not in grouped.columns:
        msg = f"Unknown count target: {target}"
        raise KeyError(msg)
    return grouped


def add_historical_lag_features(panel: pd.DataFrame, *, count_col: str) -> pd.DataFrame:
    """Add lag features using only strictly prior suburb-month periods (no leakage)."""
    out = panel.copy()
    out["prev_month_count"] = out.groupby("loc_suburb")[count_col].shift(1)
    out["seasonal_hist_mean"] = out.groupby(["loc_suburb", "crash_month"])[count_col].transform(
        lambda series: series.shift(1).expanding().mean()
    )
    out["suburb_expanding_mean"] = out.groupby("loc_suburb")[count_col].transform(
        lambda series: series.shift(1).expanding().mean()
    )
    return out


def assign_temporal_splits(panel: pd.DataFrame, splits: YearSplits) -> pd.DataFrame:
    """Attach train/val/test labels from whole-year configuration."""
    out = panel.copy()
    out["split"] = assign_split_column(out, splits)
    return out


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_clipped = np.clip(y_pred, 1e-9, None)
    y_true_nonneg = np.maximum(y_true, 0)
    try:
        return float(mean_poisson_deviance(y_true_nonneg, y_pred_clipped))
    except ValueError:
        return float("nan")


def _mase(y_true: np.ndarray, y_pred: np.ndarray, baseline_pred: np.ndarray) -> float:
    mae = mean_absolute_error(y_true, y_pred)
    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    if baseline_mae < 1e-9:
        return float("nan")
    return float(mae / baseline_mae)


def _overdispersion(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residuals = y_true - y_pred
    var = float(np.var(residuals))
    mean = float(np.mean(y_true))
    if mean < 1e-9:
        return float("nan")
    return var / mean


def _evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    seasonal_baseline: np.ndarray,
) -> dict[str, Any]:
    y_pred_nonneg = np.clip(y_pred, 0, None)
    residuals = y_true - y_pred_nonneg
    return {
        "mae": float(mean_absolute_error(y_true, y_pred_nonneg)),
        "rmse": _rmse(y_true, y_pred_nonneg),
        "poisson_deviance": _poisson_deviance(y_true, y_pred_nonneg),
        "mase_vs_seasonal": _mase(y_true, y_pred_nonneg, seasonal_baseline),
        "overdispersion": _overdispersion(y_true, y_pred_nonneg),
        "zero_frequency_actual": float(np.mean(y_true == 0)),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "n": int(len(y_true)),
    }


def predict_overall_mean(train: pd.DataFrame, test: pd.DataFrame, *, count_col: str) -> np.ndarray:
    global_mean = float(train[count_col].mean())
    return np.full(len(test), global_mean)


def predict_seasonal_mean(test: pd.DataFrame) -> np.ndarray:
    return test["seasonal_hist_mean"].fillna(0.0).to_numpy()


def predict_previous_month(test: pd.DataFrame) -> np.ndarray:
    return test["prev_month_count"].fillna(0.0).to_numpy()


def _glm_design(df: pd.DataFrame) -> pd.DataFrame:
    month_dummies = pd.get_dummies(df["crash_month"].astype(int), prefix="m", drop_first=True)
    features = pd.DataFrame(
        {
            "prev_month_count": pd.to_numeric(df["prev_month_count"], errors="coerce").fillna(0.0),
            "seasonal_hist_mean": pd.to_numeric(df["seasonal_hist_mean"], errors="coerce").fillna(
                0.0
            ),
        },
        index=df.index,
    )
    design = pd.concat([features, month_dummies], axis=1)
    return design.astype(np.float64)


def _count_design_columns(train: pd.DataFrame) -> list[str]:
    """Column schema for count GLM/HGB design matrices (fit on train only)."""
    return list(_glm_design(train).columns)


def _glm_design_aligned(df: pd.DataFrame, design_columns: list[str]) -> pd.DataFrame:
    """Align encoded count design to train-fitted column order."""
    return _glm_design(df).reindex(columns=design_columns, fill_value=0.0).astype(np.float64)


def fit_predict_poisson_glm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    count_col: str,
) -> np.ndarray:
    design_columns = _count_design_columns(train)
    x_train = sm.add_constant(
        _glm_design_aligned(train, design_columns), has_constant="add"
    ).astype(np.float64)
    x_test = sm.add_constant(_glm_design_aligned(test, design_columns), has_constant="add").astype(
        np.float64
    )
    endog = pd.to_numeric(train[count_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    model = sm.GLM(
        endog,
        x_train,
        family=sm.families.Poisson(),
    )
    result = model.fit()
    return np.asarray(result.predict(x_test))


def fit_predict_negbin_glm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    count_col: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    design_columns = _count_design_columns(train)
    x_train = sm.add_constant(
        _glm_design_aligned(train, design_columns), has_constant="add"
    ).astype(np.float64)
    x_test = sm.add_constant(_glm_design_aligned(test, design_columns), has_constant="add").astype(
        np.float64
    )
    endog = pd.to_numeric(train[count_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    from statsmodels.discrete.discrete_model import NegativeBinomial  # type: ignore[import-untyped]

    model = NegativeBinomial(endog, x_train)
    result = model.fit(disp=False, maxiter=100)
    dispersion = float(getattr(result, "params", [0])[-1]) if hasattr(result, "params") else None
    return np.asarray(result.predict(x_test)), {"dispersion_param": dispersion}


def fit_predict_hgb_poisson(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    count_col: str,
    seed: int,
    max_iter: int,
) -> np.ndarray:
    design_columns = _count_design_columns(train)
    x_train = _glm_design_aligned(train, design_columns).to_numpy(dtype=np.float64)
    x_test = _glm_design_aligned(test, design_columns).to_numpy(dtype=np.float64)
    y_train = pd.to_numeric(train[count_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=max_iter,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return np.asarray(np.clip(pred, 0, None), dtype=float)


def _run_one_model(
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    count_col: str,
    seed: int,
    max_iter: int,
) -> dict[str, Any]:
    seasonal_val = predict_seasonal_mean(val)
    seasonal_test = predict_seasonal_mean(test)

    extra: dict[str, Any] = {}
    if model_name == "overall_mean":
        val_pred = predict_overall_mean(train, val, count_col=count_col)
        test_pred = predict_overall_mean(train, test, count_col=count_col)
    elif model_name == "seasonal_mean":
        val_pred = seasonal_val
        test_pred = seasonal_test
    elif model_name == "previous_month":
        val_pred = predict_previous_month(val)
        test_pred = predict_previous_month(test)
    elif model_name == "poisson":
        val_pred = fit_predict_poisson_glm(train, val, count_col=count_col)
        test_pred = fit_predict_poisson_glm(train, test, count_col=count_col)
    elif model_name == "negbin":
        val_pred, extra = fit_predict_negbin_glm(train, val, count_col=count_col)
        test_pred, _ = fit_predict_negbin_glm(train, test, count_col=count_col)
    elif model_name == "hgb_poisson":
        val_pred = fit_predict_hgb_poisson(
            train, val, count_col=count_col, seed=seed, max_iter=max_iter
        )
        test_pred = fit_predict_hgb_poisson(
            train, test, count_col=count_col, seed=seed, max_iter=max_iter
        )
    else:
        msg = f"Unknown count model: {model_name}"
        raise ValueError(msg)

    return {
        "model_name": model_name,
        "valid": True,
        "val_metrics": _evaluate_predictions(
            val[count_col].to_numpy(),
            val_pred,
            seasonal_baseline=seasonal_val,
        ),
        "test_metrics": _evaluate_predictions(
            test[count_col].to_numpy(),
            test_pred,
            seasonal_baseline=seasonal_test,
        ),
        **extra,
    }


def run_count_training(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Train suburb-month count models with temporal holdout."""
    started = time.perf_counter()
    metrics_path = artifacts_metrics_dir(paths) / f"counts_{config.profile}.json"

    if not force and metrics_path.is_file():
        logger.info("Count model outputs exist; skipping (use --force)")
        return {"status": "skipped", "metrics_path": str(metrics_path)}

    seed = config.seed
    set_random_seed(seed)
    model_names = _configured_models(config)
    max_iter = int(config.tuning.get("n_estimators_cap", 100))

    df = _load_frame(paths, config.profile)
    panel = build_count_panel(df, target="crash_count")
    panel = add_historical_lag_features(panel, count_col="crash_count")

    years = sorted(int(y) for y in panel["crash_year"].unique())
    splits_cfg = config.raw.get("splits", {})
    splits = compute_year_splits(
        years,
        train_year_end=int(splits_cfg.get("train_year_end", 2021))
        if isinstance(splits_cfg, dict)
        else None,
        val_years=[int(y) for y in splits_cfg.get("val_years", [2022])]
        if isinstance(splits_cfg, dict) and isinstance(splits_cfg.get("val_years"), list)
        else None,
        test_years=[int(y) for y in splits_cfg.get("test_years", [2023])]
        if isinstance(splits_cfg, dict) and isinstance(splits_cfg.get("test_years"), list)
        else None,
    )
    panel = assign_temporal_splits(panel, splits)

    train = panel.loc[panel["split"] == "train"]
    val = panel.loc[panel["split"] == "val"]
    test = panel.loc[panel["split"] == "test"]
    if train.empty or val.empty or test.empty:
        msg = "Insufficient suburb-month rows for train/val/test count modelling"
        raise ValueError(msg)

    logger.info(
        "Count models on panel rows train=%d val=%d test=%d models=%s",
        len(train),
        len(val),
        len(test),
        model_names,
    )

    candidates: list[dict[str, Any]] = []
    for model_name in model_names:
        try:
            candidate = _run_one_model(
                model_name,
                train,
                val,
                test,
                count_col="crash_count",
                seed=seed,
                max_iter=max_iter,
            )
            candidates.append(candidate)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Count model %s failed: %s", model_name, exc)
            candidates.append({"model_name": model_name, "valid": False, "error": str(exc)})

    mean_variance = {
        "train_mean": float(train["crash_count"].mean()),
        "train_variance": float(train["crash_count"].var()),
        "train_dispersion_ratio": float(train["crash_count"].var() / train["crash_count"].mean())
        if train["crash_count"].mean() > 0
        else None,
    }

    elapsed = time.perf_counter() - started
    payload = {
        "task": "counts",
        "profile": config.profile,
        "timestamp_utc": utc_now_iso(),
        "seed": seed,
        "target": "crash_count",
        "n_panel_rows": len(panel),
        "splits": splits.as_dict(),
        "mean_variance": mean_variance,
        "candidates": candidates,
        "exposure_note": (
            "Count models predict recorded crashes per suburb-month; "
            "not exposure-adjusted intrinsic risk."
        ),
        "timings": {"count_training_seconds": elapsed},
    }
    persist_json(metrics_path, payload)
    logger.info("Count training finished in %.2fs", elapsed)
    return {"status": "completed", **payload}
