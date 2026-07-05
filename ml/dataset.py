from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.columns import Col, TARGET
from ml.encoding import FeatureEncoder, fit_transform, transform


def load_features(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET],
    )
    return train_df, test_df


def build_xy(
    df: pd.DataFrame,
    feature_columns: tuple[Col, ...],
    *,
    encoder: FeatureEncoder | None = None,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, FeatureEncoder]:
    if encoder is None:
        return fit_transform(df, feature_columns)
    x, y = transform(df, feature_columns, encoder)
    return x, y


def align_test_columns(x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
    return x_test.reindex(columns=x_train.columns, fill_value=0)
