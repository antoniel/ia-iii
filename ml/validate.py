from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass
class ValidateResult:
    metrics: dict[str, float]
    importances: pd.Series


def load_model(path: Path) -> RandomForestClassifier:
    return joblib.load(path)


def evaluate(model: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def feature_importances(model: RandomForestClassifier, feature_names: list[str]) -> pd.Series:
    series = pd.Series(model.feature_importances_, index=feature_names)
    return series.sort_values(ascending=False)


def run_validate(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> ValidateResult:
    metrics = evaluate(model, x_test, y_test)
    importances = feature_importances(model, feature_names)
    return ValidateResult(metrics=metrics, importances=importances)
