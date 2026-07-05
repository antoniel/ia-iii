from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Campos internos / controle do sistema (SINASC — não usar como feature)
SYSTEM_COLUMNS = {
    "CONTADOR",  # Número identificador do registro
    "STDNEPIDEM",  # Status de DN epidemiológica (0 não · 1 sim)
    "STDNNOVA",  # Status de DN nova (0 não · 1 sim)
    "DTCADASTRO",  # Data do cadastro da DN no sistema
    "DTRECEBIM",  # Data do último recebimento do lote (Sisnet)
    "DTRECORIGA",  # Data do primeiro recebimento (derivada pelo sistema)
    "DIFDATA",  # Diferença DTNASC − DTRECORIGA (controle de fluxo)
    "NUMEROLOTE",  # Número do lote de transmissão
    "VERSAOSIST",  # Versão do sistema que gerou o registro
    "ORIGEM",  # Banco de origem (1 Oracle · 2 FTP · 3 SEAD)
}

# Consequência do parto ou do processo obstétrico — vazam o alvo PARTO
LEAKAGE_COLUMNS = {
    "PESO",  # Peso ao nascer (gramas, até 5ª hora de vida)
    "APGAR1",  # Índice de Apgar no 1º minuto (00–10)
    "APGAR5",  # Índice de Apgar no 5º minuto (00–10)
    "STCESPARTO",  # Cesárea ocorreu antes do trabalho de parto iniciar?
    "STTRABPART",  # Trabalho de parto foi induzido?
}

# Datas administrativas — não entram no MVP de determinantes
METADATA_COLUMNS = {
    "DTNASC",  # Data de nascimento do recém-nascido
    "HORANASC",  # Horário do nascimento
    "DTNASCMAE",  # Data de nascimento da mãe
    "DTULTMENST",  # Data da última menstruação (DUM)
    "DTDECLARAC",  # Data do preenchimento da declaração
}


@dataclass
class PreprocessConfig:
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    glob_pattern: str = "sinasc_ba_*.parquet"
    output_path: Path = field(
        default_factory=lambda: Path("data/processed/sinasc_ba_unified.parquet")
    )
    valid_parto_only: bool = True
    drop_system: bool = True
    drop_leakage: bool = True
    drop_metadata: bool = True
    add_target: bool = True
    extra_drop_columns: tuple[str, ...] = ()


def _year_from_path(path: Path) -> int | None:
    match = re.search(r"sinasc_ba_(\d{4})\.parquet$", path.name)
    return int(match.group(1)) if match else None


def load_raw(config: PreprocessConfig) -> pd.DataFrame:
    files = sorted(config.raw_dir.glob(config.glob_pattern))
    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo em {config.raw_dir}/{config.glob_pattern}"
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        chunk = pd.read_parquet(path)
        chunk["ano"] = _year_from_path(path)
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


def filter_valid_parto(df: pd.DataFrame) -> pd.DataFrame:
    parto = df["PARTO"].astype(str).str.strip()
    return df.loc[parto.isin(["1", "2"])].copy()


def drop_columns(df: pd.DataFrame, columns: set[str] | list[str]) -> pd.DataFrame:
    to_drop = [c for c in columns if c in df.columns]
    return df.drop(columns=to_drop)


def columns_to_drop(config: PreprocessConfig) -> set[str]:
    drops = set(config.extra_drop_columns)
    if config.drop_system:
        drops |= SYSTEM_COLUMNS
    if config.drop_leakage:
        drops |= LEAKAGE_COLUMNS
    if config.drop_metadata:
        drops |= METADATA_COLUMNS
    return drops


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parto = out["PARTO"].astype(str).str.strip()
    out["y_cesarea"] = (parto == "2").astype("int8")
    return out


def run_preprocess(config: PreprocessConfig | None = None) -> pd.DataFrame:
    cfg = config or PreprocessConfig()

    raw = load_raw(cfg)

    if cfg.valid_parto_only:
        filtered = filter_valid_parto(raw)
    else:
        filtered = raw

    cleaned = drop_columns(filtered, columns_to_drop(cfg))

    if cfg.add_target:
        final = add_target(cleaned)
    else:
        final = cleaned

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(cfg.output_path, index=False)
    return final


def main() -> None:
    df = run_preprocess()
    print(f"Preprocess OK")
    print(
        f"   linhas={len(df):,}  anos={df['ano'].nunique()}  cesárea={df['y_cesarea'].mean() * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
