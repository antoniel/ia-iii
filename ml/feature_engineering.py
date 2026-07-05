from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ml.columns import COLUMN_RENAMES, DERIVED_FEATURE_COLUMNS, Col
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
    extra = (
        set(df.columns)
        - set(COLUMN_RENAMES)
        - DERIVED_FEATURE_COLUMNS
        - {Col.ANO.value, Col.Y_CESAREA.value}
    )
    if extra:
        raise KeyError(f"Colunas sem mapeamento: {sorted(extra)}")
    return df.rename(columns=COLUMN_RENAMES)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    residencia = out[Col.CODIGO_MUNICIPIO_RESIDENCIA.value].astype(str).str.strip()
    nascimento = out[Col.CODIGO_MUNICIPIO_NASCIMENTO.value].astype(str).str.strip()
    mesmo_municipio = (
        (residencia == nascimento) & (residencia != "") & (nascimento != "")
    )
    out[Col.NASCIMENTO_MESMO_MUNICIPIO_RESIDENCIA.value] = mesmo_municipio.astype("int8")
    return out


def run_features(df: pd.DataFrame, config: FeaturesConfig | None = None) -> pd.DataFrame:
    cfg = config or FeaturesConfig()
    renamed = rename_columns(df)
    features = add_derived_features(renamed)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cfg.output_path, index=False)
    return features


def main() -> None:
    preprocessed = run_preprocess()
    features = run_features(preprocessed)
    print(f"Features OK → {FeaturesConfig().output_path}")
    print(f"   linhas={len(features):,}  cols={len(features.columns)}")


if __name__ == "__main__":
    main()
