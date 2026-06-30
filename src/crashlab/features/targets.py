"""Target construction for severity modelling tasks."""

from __future__ import annotations

import pandas as pd

from crashlab.data.schema import SEVERITY_PDO, SEVERITY_VALUES

BINARY_SEVERE: frozenset[str] = frozenset({"Fatal", "Hospitalisation"})

MULTICLASS_ORDER: tuple[str, ...] = (
    "Minor Injury",
    "Medical Treatment",
    "Hospitalisation",
    "Fatal",
)

SEVERITY_TO_CLASS: dict[str, int] = {label: index for index, label in enumerate(MULTICLASS_ORDER)}


def filter_modeling_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows eligible for severity modelling (PDO already excluded upstream)."""
    if "crash_severity" not in df.columns:
        return df.copy()
    mask = df["crash_severity"].isin(SEVERITY_VALUES)
    return df.loc[mask].copy()


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``severe_binary`` (1 = Fatal/Hospitalisation, 0 otherwise)."""
    out = df.copy()
    if "crash_severity" not in out.columns:
        out["severe_binary"] = pd.NA
        return out
    out["severe_binary"] = out["crash_severity"].isin(BINARY_SEVERE).astype(int)
    return out


def add_multiclass_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``severity_class`` with order 0–3 (Minor → Fatal)."""
    out = df.copy()
    if "crash_severity" not in out.columns:
        out["severity_class"] = pd.NA
        return out
    out["severity_class"] = out["crash_severity"].map(SEVERITY_TO_CLASS)
    return out


def add_ordinal_target(df: pd.DataFrame) -> pd.DataFrame:
    """Alias ordinal target to multiclass order (cumulative models use same scale)."""
    return add_multiclass_target(df).rename(columns={"severity_class": "severity_ordinal"})


def assert_no_pdo(df: pd.DataFrame) -> None:
    """Raise when property-damage-only rows remain in a modelling frame."""
    if "crash_severity" not in df.columns:
        return
    if (df["crash_severity"] == SEVERITY_PDO).any():
        msg = "PDO rows must be excluded before target construction"
        raise ValueError(msg)
