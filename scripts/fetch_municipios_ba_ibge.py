"""Baixa municípios da Bahia (API IBGE) e salva CSV de referência."""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

import pandas as pd

IBGE_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/29/municipios"
OUT_PATH = Path("data/reference/municipios_ba_ibge.csv")


def fetch_municipios_ba() -> pd.DataFrame:
    request = urllib.request.Request(
        IBGE_URL,
        headers={"Accept": "application/json", "Accept-Encoding": "identity"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    try:
        payload = raw.decode("utf-8")
    except UnicodeDecodeError:
        payload = gzip.decompress(raw).decode("utf-8")
    data = json.loads(payload)

    rows = []
    for item in data:
        codigo_completo = str(item["id"])
        rows.append(
            {
                "codigo_municipio": codigo_completo[:6],
                "codigo_ibge_completo": codigo_completo,
                "nome_municipio": item["nome"],
            }
        )
    return pd.DataFrame(rows).drop_duplicates("codigo_municipio")


def main() -> None:
    ref = fetch_municipios_ba()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(OUT_PATH, index=False)
    print(f"{len(ref)} municípios → {OUT_PATH}")


if __name__ == "__main__":
    main()
