from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ml.columns import (
    CATEGORICAL_FEATURES,
    Col,
    TARGET,
    TARGET_ENCODE_FEATURES,
    target_encode_column_name,
)

DEFAULT_SMOOTHING = 10.0


@dataclass
class FeatureEncoder:
    """Encoders fit no treino: target encoding (alta cardinalidade) + one-hot (baixa)."""

    global_mean: float
    smoothing: float
    target_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    one_hot_columns: list[str] = field(default_factory=list)
    output_columns: list[str] = field(default_factory=list)


def _smooth_rate(count: int, category_mean: float, global_mean: float, smoothing: float) -> float:
    return (count * category_mean + smoothing * global_mean) / (count + smoothing)


def _fit_target_map(series: pd.Series, y: pd.Series, global_mean: float, smoothing: float) -> dict[str, float]:
    frame = pd.DataFrame({"cat": series.astype(str), "y": y})
    stats = frame.groupby("cat")["y"].agg(["mean", "count"])
    return {
        category: _smooth_rate(int(row["count"]), float(row["mean"]), global_mean, smoothing)
        for category, row in stats.iterrows()
    }


def _apply_target_encoding(series: pd.Series, mapping: dict[str, float], global_mean: float) -> pd.Series:
    return series.astype(str).map(mapping).fillna(global_mean)


def _numeric_columns(feature_columns: tuple[Col, ...]) -> list[Col]:
    return [
        col
        for col in feature_columns
        if col not in CATEGORICAL_FEATURES and col not in TARGET_ENCODE_FEATURES
    ]


def _one_hot_columns(feature_columns: tuple[Col, ...]) -> list[Col]:
    return [
        col
        for col in feature_columns
        if col in CATEGORICAL_FEATURES and col not in TARGET_ENCODE_FEATURES
    ]


def _target_encode_columns(feature_columns: tuple[Col, ...]) -> list[Col]:
    return [col for col in feature_columns if col in TARGET_ENCODE_FEATURES]


def fit_transform(
    df: pd.DataFrame,
    feature_columns: tuple[Col, ...],
    smoothing: float = DEFAULT_SMOOTHING,
) -> tuple[pd.DataFrame, pd.Series, FeatureEncoder]:
    y = df[TARGET]
    global_mean = float(y.mean())
    encoder = FeatureEncoder(global_mean=global_mean, smoothing=smoothing)

    parts: list[pd.DataFrame] = []

    for col in _numeric_columns(feature_columns):
        parts.append(pd.to_numeric(df[col.value], errors="coerce").rename(col.value).to_frame())

    for col in _target_encode_columns(feature_columns):
        encoder.target_maps[col.value] = _fit_target_map(df[col.value], y, global_mean, smoothing)
        encoded_name = target_encode_column_name(col)
        parts.append(
            _apply_target_encoding(
                df[col.value],
                encoder.target_maps[col.value],
                global_mean,
            ).rename(encoded_name).to_frame()
        )

    one_hot_cols = _one_hot_columns(feature_columns)
    if one_hot_cols:
        categorical = df[[c.value for c in one_hot_cols]].astype(str)
        one_hot = pd.get_dummies(categorical, columns=[c.value for c in one_hot_cols], dtype=float)
        encoder.one_hot_columns = list(one_hot.columns)
        parts.append(one_hot)

    x = pd.concat(parts, axis=1)
    medians = x.median(numeric_only=True)
    x = x.fillna(medians).fillna(0)
    encoder.output_columns = list(x.columns)
    return x, y, encoder


def transform(
    df: pd.DataFrame,
    feature_columns: tuple[Col, ...],
    encoder: FeatureEncoder,
) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET]
    parts: list[pd.DataFrame] = []

    for col in _numeric_columns(feature_columns):
        parts.append(pd.to_numeric(df[col.value], errors="coerce").rename(col.value).to_frame())

    for col in _target_encode_columns(feature_columns):
        encoded_name = target_encode_column_name(col)
        mapping = encoder.target_maps[col.value]
        parts.append(
            _apply_target_encoding(df[col.value], mapping, encoder.global_mean)
            .rename(encoded_name)
            .to_frame()
        )

    one_hot_cols = _one_hot_columns(feature_columns)
    if one_hot_cols:
        categorical = df[[c.value for c in one_hot_cols]].astype(str)
        one_hot = pd.get_dummies(categorical, columns=[c.value for c in one_hot_cols], dtype=float)
        one_hot = one_hot.reindex(columns=encoder.one_hot_columns, fill_value=0.0)
        parts.append(one_hot)

    x = pd.concat(parts, axis=1)
    x = x.reindex(columns=encoder.output_columns, fill_value=0.0)
    medians = x.median(numeric_only=True)
    x = x.fillna(medians).fillna(0)
    return x, y
