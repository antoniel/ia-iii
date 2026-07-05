from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ml.columns import COLUMN_RENAMES, Col
from ml.preprocess import run_preprocess


@dataclass
class FeaturesConfig:
    output_path: Path = field(
        default_factory=lambda: Path("data/processed/sinasc_ba_features.parquet")
    )


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMN_RENAMES) - set(df.columns)
    if missing:
        raise KeyError(f"Colunas ausentes para rename: {sorted(missing)}")
    extra = set(df.columns) - set(COLUMN_RENAMES) - {Col.ANO.value, Col.Y_CESAREA.value}
    if extra:
        raise KeyError(f"Colunas sem mapeamento: {sorted(extra)}")
    return df.rename(columns=COLUMN_RENAMES)


def run_features(df: pd.DataFrame, config: FeaturesConfig | None = None) -> pd.DataFrame:
    cfg = config or FeaturesConfig()
    renamed = rename_columns(df)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    renamed.to_parquet(cfg.output_path, index=False)
    return renamed


def main() -> None:
    preprocessed = run_preprocess()
    features = run_features(preprocessed)
    print(f"Features OK → {FeaturesConfig().output_path}")
    print(f"   linhas={len(features):,}  cols={len(features.columns)}")


if __name__ == "__main__":
    main()
