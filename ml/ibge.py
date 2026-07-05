from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_MUNICIPIOS_PATH = Path("data/reference/municipios_ba_ibge.csv")


def load_municipios_ba(path: Path = DEFAULT_MUNICIPIOS_PATH) -> pd.DataFrame:
    """Mapa código IBGE (6 dígitos, SINASC) → nome do município."""
    if not path.exists():
        raise FileNotFoundError(
            f"Referência IBGE não encontrada: {path}. "
            "Execute: uv run python scripts/fetch_municipios_ba_ibge.py"
        )
    ref = pd.read_csv(path, dtype={"codigo_municipio": str, "codigo_ibge_completo": str})
    ref["codigo_municipio"] = ref["codigo_municipio"].str.strip()
    return ref


def attach_municipio_nomes(
    df: pd.DataFrame,
    *,
    codigo_col: str = "municipio",
    path: Path = DEFAULT_MUNICIPIOS_PATH,
) -> pd.DataFrame:
    """Junta nomes IBGE; códigos desconhecidos viram 'Código {code}'."""
    ref = load_municipios_ba(path)
    out = df.copy()
    out[codigo_col] = out[codigo_col].astype(str).str.strip()
    out = out.merge(ref[["codigo_municipio", "nome_municipio"]], left_on=codigo_col, right_on="codigo_municipio", how="left")
    out.loc[out[codigo_col] == "290000", "nome_municipio"] = "Não informado"
    out["nome_municipio"] = out["nome_municipio"].fillna("Código " + out[codigo_col])
    return out.drop(columns=["codigo_municipio"], errors="ignore")


def municipio_label(nome: str, *, n: int | None = None) -> str:
    if n is not None:
        return f"{nome} (n={n:,})"
    return nome
