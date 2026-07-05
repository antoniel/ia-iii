from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ml.columns import FEATURES_V0, FEATURES_V8, Col
from ml.dataset import build_xy, load_features, split_train_test
from ml.experiments import format_log_summary, log_experiment
from ml.feature_engineering import run_features
from ml.preprocess import run_preprocess
from ml.train import TrainConfig, run_train
from ml.validate import run_cross_validate, run_validate


@dataclass
class RunConfig:
    input_path: Path = field(
        default_factory=lambda: Path("data/processed/sinasc_ba_features.parquet")
    )
    model_path: Path = field(
        default_factory=lambda: Path("data/processed/model_rf_v0.joblib")
    )
    feature_columns: tuple[Col, ...] = FEATURES_V0
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 250
    max_depth: int | None = 30
    min_samples_split: int = 20
    min_samples_leaf: int = 2
    max_features: str | float = 0.3
    n_jobs: int = -1
    n_folds: int = 0
    log_experiments: bool = True
    experiments_path: Path = field(
        default_factory=lambda: Path("experiments/experiments.jsonl")
    )
    experiment_tag: str = ""
    experiment_notes: str = ""


def run(config: RunConfig | None = None):
    cfg = config or RunConfig()

    if cfg.input_path.exists():
        df = load_features(cfg.input_path)
    else:
        df = run_features(run_preprocess())

    train_cfg = TrainConfig(
        model_path=cfg.model_path,
        feature_columns=cfg.feature_columns,
        random_state=cfg.random_state,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_split=cfg.min_samples_split,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features,
        n_jobs=cfg.n_jobs,
    )

    cv_result = None
    if cfg.n_folds >= 2:
        cv_result = run_cross_validate(
            df,
            cfg.feature_columns,
            train_cfg,
            n_folds=cfg.n_folds,
            random_state=cfg.random_state,
        )

    train_df, test_df = split_train_test(df, cfg.test_size, cfg.random_state)

    x_train, y_train, encoder = build_xy(train_df, cfg.feature_columns)
    x_test, y_test = build_xy(test_df, cfg.feature_columns, encoder=encoder)

    model = run_train(x_train, y_train, train_cfg)

    result = run_validate(
        model,
        x_test,
        y_test,
        list(x_train.columns),
        cv=cv_result,
    )

    experiment = None
    if cfg.log_experiments:
        experiment = log_experiment(
            cfg,
            result,
            path=cfg.experiments_path,
            tag=cfg.experiment_tag,
            notes=cfg.experiment_notes,
            n_train=len(x_train),
            n_test=len(x_test),
            n_features=x_train.shape[1],
        )

    return model, result, experiment


def main() -> None:
    cfg = RunConfig(
        feature_columns=FEATURES_V8,
        model_path=Path("data/processed/model_rf_v8.joblib"),
        experiment_tag="v8",
        experiment_notes=(
            "v8: +partos vaginais anteriores, mesmo município nasc/residência, "
            "estado civil, sexo RN, raça/cor RN (sem TPNASCASSI)"
        ),
    )
    _, result, experiment = run(cfg)
    print(f"Model OK → {cfg.model_path}")

    if result.cv is not None:
        print(f"Cross-validation ({result.cv.n_folds}-fold, stratified):")
        for name in ("roc_auc", "f1", "accuracy"):
            mean = result.cv.metrics_mean[name]
            std = result.cv.metrics_std[name]
            print(f"   {name}={mean:.3f} ± {std:.3f}")

    print("Hold-out (20%):")
    print(f"   accuracy={result.metrics['accuracy']:.3f}")
    print(f"   f1={result.metrics['f1']:.3f}")
    print(f"   roc_auc={result.metrics['roc_auc']:.3f}")
    print("   top importances:")
    for name, score in result.importances.head(5).items():
        print(f"      {name}: {score:.4f}")
    if experiment is not None:
        print(format_log_summary(experiment, cfg.experiments_path))


if __name__ == "__main__":
    main()
