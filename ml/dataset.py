from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.columns import CATEGORICAL_FEATURES, Col, TARGET


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


def build_xy(df: pd.DataFrame, feature_columns: tuple[Col, ...]) -> tuple[pd.DataFrame, pd.Series]:
    subset = df[list(feature_columns)].copy()
    y = df[TARGET]

    numeric = [c for c in feature_columns if c not in CATEGORICAL_FEATURES]
    categorical = [c for c in feature_columns if c in CATEGORICAL_FEATURES]

    for col in numeric:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    if categorical:
        for col in categorical:
            subset[col] = subset[col].astype(str)
        x = pd.get_dummies(subset, columns=categorical, dtype=int)
    else:
        x = subset

    medians = x.median(numeric_only=True)
    x = x.fillna(medians).fillna(0)
    return x, y


def align_test_columns(x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
    return x_test.reindex(columns=x_train.columns, fill_value=0)
