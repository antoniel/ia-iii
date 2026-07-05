from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.columns import FEATURES_V0, Col


@dataclass
class TrainConfig:
    model_path: Path = field(
        default_factory=lambda: Path("data/processed/model_rf_v0.joblib")
    )
    feature_columns: tuple[Col, ...] = FEATURES_V0
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int | None = None
    n_jobs: int = -2


def train_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainConfig,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    return model


def run_train(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainConfig | None = None,
) -> RandomForestClassifier:
    cfg = config or TrainConfig()
    model = train_random_forest(x_train, y_train, cfg)

    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.model_path)
    return model
