"""Train-only categorical encoders with min-frequency filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from sklearn.impute import SimpleImputer as SimpleImputerType


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
    imputer_: SimpleImputerType | None = None
    feature_columns_: tuple[str, ...] = ()

    def feature_names(self) -> list[str]:
        names: list[str] = list(self.numeric_columns)
        for encoder in self.encoders.values():
            if self.encoding == "frequency":
                names.append(f"{encoder.column}__freq")
            else:
                names.extend(f"{encoder.column}__{cat}" for cat in encoder.categories)
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return transform_with_mixed_encoding(self, df)


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


def _encode_dataframe(
    bundle: EncoderBundle,
    df: pd.DataFrame,
    *,
    high_cardinality_threshold: int = 30,
) -> pd.DataFrame:
    """Encode categoricals and numerics without imputation."""
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


def _align_feature_columns(raw: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """Ensure encoded matrix column order matches the train-fitted schema."""
    aligned = raw.reindex(columns=list(feature_columns), fill_value=0.0)
    return aligned.astype(np.float64)


def _apply_imputer(
    bundle: EncoderBundle,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """Apply train-fitted imputer; fall back to zero-fill for legacy bundles."""
    if bundle.feature_columns_:
        raw = _align_feature_columns(raw, bundle.feature_columns_)
    else:
        raw = raw.astype(np.float64)

    if bundle.imputer_ is not None:
        imputed = bundle.imputer_.transform(raw)
        imputed = np.nan_to_num(imputed, nan=0.0, posinf=0.0, neginf=0.0)
        columns = list(bundle.feature_columns_) or list(raw.columns)
        return pd.DataFrame(imputed, columns=columns, index=raw.index)

    filled = raw.fillna(0.0)
    return filled.astype(np.float64)


def fit_imputer_on_train(bundle: EncoderBundle, train_df: pd.DataFrame) -> EncoderBundle:
    """Fit median imputer on training-encoded features only."""
    raw_train = _encode_dataframe(bundle, train_df)
    if raw_train.empty:
        return bundle
    raw_train = raw_train.astype(np.float64)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(raw_train)
    bundle.imputer_ = imputer
    bundle.feature_columns_ = tuple(raw_train.columns)
    return bundle


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
            pass
    bundle_encoding = encoding
    if any(
        train_df[col].nunique(dropna=True) > high_cardinality_threshold
        for col in categorical_columns
        if col in train_df.columns
    ):
        bundle_encoding = "mixed"

    bundle = EncoderBundle(
        encoders=encoders,
        numeric_columns=tuple(numeric_columns),
        encoding=bundle_encoding,
    )
    return fit_imputer_on_train(bundle, train_df)


def transform_with_mixed_encoding(
    bundle: EncoderBundle,
    df: pd.DataFrame,
    *,
    high_cardinality_threshold: int = 30,
) -> pd.DataFrame:
    """Transform using one-hot or frequency per column based on train cardinality."""
    raw = _encode_dataframe(bundle, df, high_cardinality_threshold=high_cardinality_threshold)
    return _apply_imputer(bundle, raw)
