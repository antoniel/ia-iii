from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ml.columns import Col, TARGET
from ml.dataset import build_xy
from ml.train import TrainConfig, train_random_forest


@dataclass
class CrossValidateResult:
    n_folds: int
    fold_metrics: list[dict[str, float]]
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]


@dataclass
class ValidateResult:
    metrics: dict[str, float]
    importances: pd.Series
    cv: CrossValidateResult | None = None


def load_model(path: Path) -> RandomForestClassifier:
    return joblib.load(path)


def evaluate(
    model: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def aggregate_metrics(fold_metrics: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = fold_metrics[0].keys()
    mean = {key: float(np.mean([fold[key] for fold in fold_metrics])) for key in keys}
    std = {
        key: float(np.std([fold[key] for fold in fold_metrics], ddof=1))
        if len(fold_metrics) > 1
        else 0.0
        for key in keys
    }
    return mean, std


def run_cross_validate(
    df: pd.DataFrame,
    feature_columns: tuple[Col, ...],
    train_config: TrainConfig,
    n_folds: int = 5,
    random_state: int = 42,
) -> CrossValidateResult:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_metrics: list[dict[str, float]] = []

    for train_idx, val_idx in skf.split(df, df[TARGET]):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        x_train, y_train, encoder = build_xy(train_df, feature_columns)
        x_val, y_val = build_xy(val_df, feature_columns, encoder=encoder)

        model = train_random_forest(x_train, y_train, train_config)
        fold_metrics.append(evaluate(model, x_val, y_val))

    metrics_mean, metrics_std = aggregate_metrics(fold_metrics)
    return CrossValidateResult(
        n_folds=n_folds,
        fold_metrics=fold_metrics,
        metrics_mean=metrics_mean,
        metrics_std=metrics_std,
    )


def feature_importances(model: RandomForestClassifier, feature_names: list[str]) -> pd.Series:
    series = pd.Series(model.feature_importances_, index=feature_names)
    return series.sort_values(ascending=False)


def run_validate(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    *,
    cv: CrossValidateResult | None = None,
) -> ValidateResult:
    metrics = evaluate(model, x_test, y_test)
    importances = feature_importances(model, feature_names)
    return ValidateResult(metrics=metrics, importances=importances, cv=cv)
