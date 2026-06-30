"""Reproducible exploratory data analysis tables and figures."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from crashlab.config import CrashlabConfig
from crashlab.data.artifacts import processed_path
from crashlab.data.manifest import utc_now_iso
from crashlab.features.spatial import add_spatial_cell, spatial_coverage_summary
from crashlab.features.targets import BINARY_SEVERE, filter_modeling_rows
from crashlab.features.temporal import YearSplits, assign_split_column, compute_year_splits
from crashlab.logging import get_logger
from crashlab.paths import CrashlabPaths

logger = get_logger("evaluation.eda")

SOURCE_NOTE = "Queensland road crash data (Brisbane City); cleaned Parquet"
PERIOD_NOTE = "Excludes property-damage-only records"


def _figure_path(paths: CrashlabPaths, name: str) -> Path:
    return paths.figures_dir / name


def _table_path(paths: CrashlabPaths, name: str) -> Path:
    return paths.tables_dir / name


def _styled_bar(
    series: pd.Series,
    *,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    series.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(f"{title}\n{PERIOD_NOTE}")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    fig.text(0.01, 0.01, SOURCE_NOTE, fontsize=7, color="gray")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def rows_by_year_severity(df: pd.DataFrame, paths: CrashlabPaths) -> pd.DataFrame:
    table = (
        df.groupby(["crash_year", "crash_severity"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["crash_year", "crash_severity"])
    )
    out = _table_path(paths, "rows_by_year_severity.csv")
    table.to_csv(out, index=False)
    pivot = table.pivot(index="crash_year", columns="crash_severity", values="count").fillna(0)
    _styled_bar(
        pivot.sum(axis=1),
        title="Crash rows by year",
        ylabel="Count (all severities)",
        path=_figure_path(paths, "rows_by_year.png"),
    )
    return table


def severe_prevalence_by_year(df: pd.DataFrame, paths: CrashlabPaths) -> pd.DataFrame:
    working = df.copy()
    working["is_severe"] = working["crash_severity"].isin(BINARY_SEVERE).astype(int)
    grouped = working.groupby("crash_year").agg(
        total=("crash_severity", "size"),
        severe=("is_severe", "sum"),
    )
    grouped["severe_rate"] = grouped["severe"] / grouped["total"]
    out = _table_path(paths, "severe_prevalence_by_year.csv")
    grouped.reset_index().to_csv(out, index=False)
    _styled_bar(
        grouped["severe_rate"],
        title="Severe crash prevalence by year",
        ylabel="Proportion severe (denom = all injury crashes)",
        path=_figure_path(paths, "severe_prevalence_by_year.png"),
    )
    return grouped.reset_index()


def missingness_table(df: pd.DataFrame, paths: CrashlabPaths) -> pd.DataFrame:
    rates = df.isna().mean().sort_values(ascending=False)
    table = rates.reset_index()
    table.columns = ["column", "missing_rate"]
    table.to_csv(_table_path(paths, "missingness.csv"), index=False)
    top = rates.head(15)
    _styled_bar(
        top,
        title="Top missingness by column",
        ylabel="Missing rate",
        path=_figure_path(paths, "missingness_top.png"),
    )
    return table


def cardinality_table(df: pd.DataFrame, paths: CrashlabPaths) -> pd.DataFrame:
    rows = [
        {"column": col, "nunique": int(df[col].nunique(dropna=True))}
        for col in sorted(df.columns)
        if df[col].dtype == object or str(df[col].dtype) == "category"
    ]
    table = pd.DataFrame(rows).sort_values("nunique", ascending=False)
    table.to_csv(_table_path(paths, "category_cardinality.csv"), index=False)
    return table


def distribution_by_hour(df: pd.DataFrame, paths: CrashlabPaths) -> None:
    counts = df["crash_hour"].value_counts().sort_index()
    _styled_bar(
        counts,
        title="Crashes by hour of day",
        ylabel="Count",
        path=_figure_path(paths, "crashes_by_hour.png"),
    )


def split_shift_table(df: pd.DataFrame, splits: YearSplits, paths: CrashlabPaths) -> pd.DataFrame:
    working = df.copy()
    working["split"] = assign_split_column(working, splits)
    rows = []
    for split_name in ("train", "val", "test"):
        subset = working.loc[working["split"] == split_name]
        if subset.empty:
            continue
        severe_rate = float(subset["crash_severity"].isin(BINARY_SEVERE).mean())
        rows.append(
            {
                "split": split_name,
                "rows": len(subset),
                "severe_rate": severe_rate,
                "years": sorted(int(y) for y in subset["crash_year"].dropna().unique()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(_table_path(paths, "split_shift_summary.csv"), index=False)
    if len(table):
        _styled_bar(
            table.set_index("split")["severe_rate"],
            title="Severe rate by temporal split",
            ylabel="Proportion severe",
            path=_figure_path(paths, "split_severe_rate.png"),
        )
    return table


def fatal_by_year_warning(df: pd.DataFrame, paths: CrashlabPaths) -> pd.DataFrame:
    fatal = df.loc[df["crash_severity"] == "Fatal"].groupby("crash_year").size()
    table = fatal.reset_index(name="fatal_count")
    table.to_csv(_table_path(paths, "fatal_by_year.csv"), index=False)
    if len(table):
        _styled_bar(
            table.set_index("crash_year")["fatal_count"],
            title="Fatal crashes by year (small counts — interpret cautiously)",
            ylabel="Fatal count",
            path=_figure_path(paths, "fatal_by_year.png"),
        )
    return table


def run_eda(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    force: bool = False,
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Generate EDA tables and figures from cleaned Parquet."""
    started = time.perf_counter()
    marker = paths.manifests_dir / f"eda_{config.profile}.json"
    if not force and marker.is_file():
        logger.info("EDA artifacts exist; skipping (use --force)")
        return {"status": "skipped", "manifest": str(marker)}

    if df is None:
        parquet = processed_path(paths)
        if not parquet.is_file():
            msg = f"Cleaned parquet required for EDA: {parquet}"
            raise FileNotFoundError(msg)
        df = pd.read_parquet(parquet)

    modeling = filter_modeling_rows(df)
    modeling = add_spatial_cell(modeling)

    splits_cfg = config.raw.get("splits", {})
    years = sorted(int(y) for y in modeling["crash_year"].dropna().unique())
    train_end = splits_cfg.get("train_year_end") if isinstance(splits_cfg, dict) else None
    val_years = splits_cfg.get("val_years") if isinstance(splits_cfg, dict) else None
    test_years = splits_cfg.get("test_years") if isinstance(splits_cfg, dict) else None
    year_splits = compute_year_splits(
        years,
        train_year_end=int(train_end) if train_end is not None else None,
        val_years=[int(y) for y in val_years] if isinstance(val_years, list) else None,
        test_years=[int(y) for y in test_years] if isinstance(test_years, list) else None,
    )

    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)

    rows_by_year_severity(modeling, paths)
    severe_prevalence_by_year(modeling, paths)
    missingness_table(modeling, paths)
    cardinality_table(modeling, paths)
    distribution_by_hour(modeling, paths)
    split_shift_table(modeling, year_splits, paths)
    fatal_by_year_warning(modeling, paths)
    coverage = spatial_coverage_summary(modeling)

    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": "1",
        "manifest_type": "eda",
        "timestamp_utc": utc_now_iso(),
        "profile": config.profile,
        "row_count": len(modeling),
        "spatial_coverage": coverage,
        "timings": {"eda_seconds": elapsed},
    }
    with marker.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logger.info("EDA complete in %.2fs (%d rows)", elapsed, len(modeling))
    return {
        "status": "completed",
        "manifest": str(marker),
        "row_count": len(modeling),
        "timings": {"eda_seconds": elapsed},
    }
