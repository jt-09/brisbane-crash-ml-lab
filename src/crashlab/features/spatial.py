"""Coarse spatial features without street-level identifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_GRID_SIZE: float = 0.01


def add_spatial_cell(
    df: pd.DataFrame,
    *,
    lon_col: str = "crash_longitude",
    lat_col: str = "crash_latitude",
    grid_size: float = DEFAULT_GRID_SIZE,
) -> pd.DataFrame:
    """Assign deterministic coarse grid cells from rounded coordinates."""
    out = df.copy()
    lon_series = out[lon_col] if lon_col in out.columns else pd.Series(np.nan, index=out.index)
    lat_series = out[lat_col] if lat_col in out.columns else pd.Series(np.nan, index=out.index)
    lon = pd.to_numeric(lon_series, errors="coerce")
    lat = pd.to_numeric(lat_series, errors="coerce")

    lon_bin = (lon / grid_size).round().astype("Int64")
    lat_bin = (lat / grid_size).round().astype("Int64")
    out["spatial_cell"] = lon_bin.astype(str).fillna("na") + "_" + lat_bin.astype(str).fillna("na")
    return out


def spatial_coverage_summary(df: pd.DataFrame) -> dict[str, float]:
    """Return coordinate coverage statistics for EDA tables."""
    lon_series = (
        df["crash_longitude"]
        if "crash_longitude" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    lat_series = (
        df["crash_latitude"]
        if "crash_latitude" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    lon = pd.to_numeric(lon_series, errors="coerce")
    lat = pd.to_numeric(lat_series, errors="coerce")
    valid = lon.notna() & lat.notna()
    return {
        "coordinate_coverage_rate": float(valid.mean()) if len(df) else 0.0,
        "unique_spatial_cells": int(df.get("spatial_cell", pd.Series(dtype=str)).nunique()),
    }
