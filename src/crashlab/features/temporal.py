"""Whole-year temporal splits and cyclic time features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class YearSplits:
    """Mutually exclusive whole-year partition."""

    train_years: frozenset[int]
    val_years: frozenset[int]
    test_years: frozenset[int]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train": sorted(self.train_years),
            "val": sorted(self.val_years),
            "test": sorted(self.test_years),
        }

    def validate_disjoint(self) -> None:
        if self.train_years & self.val_years:
            msg = "Train and validation years overlap"
            raise ValueError(msg)
        if self.train_years & self.test_years:
            msg = "Train and test years overlap"
            raise ValueError(msg)
        if self.val_years & self.test_years:
            msg = "Validation and test years overlap"
            raise ValueError(msg)


def compute_year_splits(
    available_years: list[int],
    *,
    train_year_end: int | None = None,
    val_years: list[int] | None = None,
    test_years: list[int] | None = None,
) -> YearSplits:
    """Derive whole-year splits from config or proportional allocation."""
    years = sorted({int(y) for y in available_years})
    if not years:
        msg = "No years available for temporal split"
        raise ValueError(msg)

    if train_year_end is not None and val_years is not None and test_years is not None:
        train = frozenset(y for y in years if y <= train_year_end)
        val = frozenset(y for y in years if y in val_years)
        test = frozenset(y for y in years if y in test_years)
        splits = YearSplits(train_years=train, val_years=val, test_years=test)
        splits.validate_disjoint()
        return splits

    n = len(years)
    n_train = max(1, math.floor(n * 0.65))
    n_val = max(1, math.floor(n * 0.20)) if n - n_train > 1 else 0
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1

    train_set = frozenset(years[:n_train])
    val_set = frozenset(years[n_train : n_train + n_val]) if n_val else frozenset()
    test_set = frozenset(years[n_train + n_val :])
    splits = YearSplits(train_years=train_set, val_years=val_set, test_years=test_set)
    splits.validate_disjoint()
    return splits


def assign_split_column(df: pd.DataFrame, splits: YearSplits) -> pd.Series:
    """Map each row to train/val/test based on ``crash_year``."""
    if "crash_year" not in df.columns:
        msg = "crash_year required for temporal split assignment"
        raise KeyError(msg)

    def _split(year: object) -> str | None:
        if year is None or (isinstance(year, float) and np.isnan(year)):
            return None
        y = int(float(str(year)))
        if y in splits.train_years:
            return "train"
        if y in splits.val_years:
            return "val"
        if y in splits.test_years:
            return "test"
        return None

    return df["crash_year"].map(_split)


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic hour/month and calendar indicator columns."""
    out = df.copy()
    hour_series = (
        out["crash_hour"] if "crash_hour" in out.columns else pd.Series(0, index=out.index)
    )
    month_series = (
        out["crash_month"] if "crash_month" in out.columns else pd.Series(1, index=out.index)
    )
    hour = pd.to_numeric(hour_series, errors="coerce").fillna(0)
    month = pd.to_numeric(month_series, errors="coerce").fillna(1)

    out["crash_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["crash_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["crash_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["crash_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    dow = out.get("crash_day_of_week")
    if dow is not None:
        weekend_days = {"Saturday", "Sunday"}
        out["is_weekend"] = dow.astype(str).isin(weekend_days).astype(int)
    else:
        out["is_weekend"] = 0

    out["is_night"] = ((hour < 6) | (hour >= 20)).astype(int)
    out["is_dawn_dusk"] = ((hour >= 5) & (hour <= 7) | (hour >= 17) & (hour <= 19)).astype(int)
    out["is_peak_hour"] = (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 18))).astype(int)
    out["is_school_commute"] = (((hour >= 7) & (hour <= 9)) | ((hour >= 14) & (hour <= 16))).astype(
        int
    )
    return out


def add_speed_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket speed limits for low-cardinality encoding."""
    out = df.copy()
    speed_series = (
        out["crash_speed_limit"]
        if "crash_speed_limit" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    speed = pd.to_numeric(speed_series, errors="coerce")

    def _bucket(value: float) -> str:
        if np.isnan(value):
            return "unknown"
        if value <= 50:
            return "le50"
        if value <= 60:
            return "51_60"
        if value <= 70:
            return "61_70"
        if value <= 80:
            return "71_80"
        return "gt80"

    out["speed_bucket"] = speed.map(_bucket)
    return out
