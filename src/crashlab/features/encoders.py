"""Train-only categorical encoders with min-frequency filtering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CategoryEncoderState:
    """Fitted mapping for one categorical column."""

    column: str
    categories: list[str]
    min_count: int
    unknown_token: str = "__unknown__"
    rare_token: str = "__rare__"
    frequency_map: dict[str, float] = field(default_factory=dict)

    def transform_one_hot(self, series: pd.Series) -> pd.DataFrame:
        values = series.fillna(self.unknown_token).astype(str)
        mapped = values.map(
            lambda v: v if v in self.categories else (self.rare_token if v else self.unknown_token)
        )
        columns = [f"{self.column}__{cat}" for cat in self.categories]
        data = np.zeros((len(mapped), len(self.categories)), dtype=np.float32)
        cat_index = {cat: idx for idx, cat in enumerate(self.categories)}
        for row_idx, cat in enumerate(mapped):
            idx = cat_index.get(cat)
            if idx is not None:
                data[row_idx, idx] = 1.0
        return pd.DataFrame(data, columns=columns, index=series.index)

    def transform_frequency(self, series: pd.Series) -> pd.Series:
        values = series.fillna(self.unknown_token).astype(str)
        freq = values.map(self.frequency_map).fillna(0.0)
        return freq.rename(f"{self.column}__freq")


@dataclass
class EncoderBundle:
    """Collection of per-column encoders fit on training data only."""

    encoders: dict[str, CategoryEncoderState] = field(default_factory=dict)
    numeric_columns: tuple[str, ...] = ()
    encoding: str = "one_hot"

    def feature_names(self) -> list[str]:
        names: list[str] = list(self.numeric_columns)
        for encoder in self.encoders.values():
            if self.encoding == "frequency":
                names.append(f"{encoder.column}__freq")
            else:
                names.extend(f"{encoder.column}__{cat}" for cat in encoder.categories)
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for col in self.numeric_columns:
            if col in df.columns:
                parts.append(pd.to_numeric(df[col], errors="coerce").to_frame(name=col))
        for col, encoder in self.encoders.items():
            if col not in df.columns:
                continue
            if self.encoding == "frequency":
                parts.append(encoder.transform_frequency(df[col]).to_frame())
            else:
                parts.append(encoder.transform_one_hot(df[col]))
        if not parts:
            return pd.DataFrame(index=df.index)
        return pd.concat(parts, axis=1)


def _fit_category_encoder(
    series: pd.Series,
    *,
    column: str,
    min_count: int,
) -> CategoryEncoderState:
    counts = series.fillna("__missing__").astype(str).value_counts()
    kept = [str(cat) for cat, count in counts.items() if count >= min_count]
    frequency_map = {str(k): float(v) for k, v in (counts / counts.sum()).to_dict().items()}
    return CategoryEncoderState(
        column=column,
        categories=kept,
        min_count=min_count,
        frequency_map=frequency_map,
    )


def fit_encoder_bundle(
    train_df: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    *,
    min_count: int = 5,
    encoding: str = "one_hot",
    high_cardinality_threshold: int = 30,
) -> EncoderBundle:
    """Fit encoders on training rows; use frequency encoding for high-cardinality cols."""
    encoders: dict[str, CategoryEncoderState] = {}
    chosen_encoding = encoding
    for col in categorical_columns:
        if col not in train_df.columns:
            continue
        nunique = train_df[col].nunique(dropna=True)
        col_encoding = "frequency" if nunique > high_cardinality_threshold else chosen_encoding
        encoder = _fit_category_encoder(train_df[col], column=col, min_count=min_count)
        encoders[col] = encoder
        if col_encoding == "frequency":
            # Re-fit with frequency path marker via categories unchanged
            pass
    bundle_encoding = encoding
    if any(
        train_df[col].nunique(dropna=True) > high_cardinality_threshold
        for col in categorical_columns
        if col in train_df.columns
    ):
        bundle_encoding = "mixed"

    return EncoderBundle(
        encoders=encoders,
        numeric_columns=tuple(numeric_columns),
        encoding=bundle_encoding,
    )


def transform_with_mixed_encoding(
    bundle: EncoderBundle,
    df: pd.DataFrame,
    *,
    high_cardinality_threshold: int = 30,
) -> pd.DataFrame:
    """Transform using one-hot or frequency per column based on train cardinality."""
    parts: list[pd.DataFrame] = []
    for col in bundle.numeric_columns:
        if col in df.columns:
            parts.append(pd.to_numeric(df[col], errors="coerce").to_frame(name=col))
    for col, encoder in bundle.encoders.items():
        if col not in df.columns:
            continue
        use_freq = len(encoder.categories) > high_cardinality_threshold or col in {
            "loc_suburb",
            "loc_abs_statistical_area_2",
        }
        if use_freq:
            parts.append(encoder.transform_frequency(df[col]).to_frame())
        else:
            parts.append(encoder.transform_one_hot(df[col]))
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)
