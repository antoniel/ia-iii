from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

COLUNA_ALVO = "parto_cesareo"

RENOMEAR = {
    "PARTO": "parto",
    "IDADEMAE": "idade_mae",
    "ESTCIVMAE": "estado_civil_mae",
    "ESCMAE2010": "escolaridade_mae",
    "RACACORMAE": "raca_cor_mae",
    "IDADEPAI": "idade_pai",
    "QTDGESTANT": "gestacoes_anteriores",
    "QTDPARTNOR": "partos_vaginais_previos",
    "QTDPARTCES": "cesareas_previas",
    "QTDFILVIVO": "filhos_vivos",
    "QTDFILMORT": "perdas_fetais",
    "GRAVIDEZ": "tipo_gravidez",
    "SEMAGESTAC": "semanas_gestacao",
    "CONSPRENAT": "consultas_prenatal",
    "MESPRENAT": "mes_inicio_prenatal",
    "TPAPRESENT": "apresentacao_fetal",
}

ATRIBUTOS_NUMERICOS = [
    "idade_mae",
    "idade_pai",
    "gestacoes_anteriores",
    "partos_vaginais_previos",
    "cesareas_previas",
    "filhos_vivos",
    "perdas_fetais",
    "semanas_gestacao",
    "consultas_prenatal",
    "mes_inicio_prenatal",
]

ATRIBUTOS_CATEGORICOS = [
    "estado_civil_mae",
    "escolaridade_mae",
    "raca_cor_mae",
    "tipo_gravidez",
    "apresentacao_fetal",
]

ATRIBUTOS_DERIVADOS = [
    "idade_pai_ausente",
    "gravidez_multipla",
    "cesarea_previa",
    "parto_vaginal_previo",
    "perda_fetal_previa",
    "inicio_prenatal_tardio",
    "apresentacao_nao_cefalica",
    "pre_termo",
]

ATRIBUTOS_MODELO = ATRIBUTOS_NUMERICOS + ATRIBUTOS_CATEGORICOS + ATRIBUTOS_DERIVADOS

NULOS = {
    "parto": {"", "9"},
    "estado_civil_mae": {"", "9"},
    "escolaridade_mae": {"", "9"},
    "raca_cor_mae": {""},
    "idade_pai": {""},
    "gestacoes_anteriores": {"", "99"},
    "partos_vaginais_previos": {"", "99"},
    "cesareas_previas": {"", "99"},
    "filhos_vivos": {"", "99"},
    "perdas_fetais": {"", "99"},
    "tipo_gravidez": {"", "9"},
    "semanas_gestacao": {""},
    "consultas_prenatal": {""},
    "mes_inicio_prenatal": {"", "99"},
    "apresentacao_fetal": {"", "9"},
}


def _ano_do_arquivo(caminho: str | Path) -> int:
    trecho = re.search(r"(20\d{2})", Path(caminho).name)
    if trecho is None:
        raise ValueError(f"Ano nao encontrado em {caminho}")
    return int(trecho.group(1))


def _texto(serie: pd.Series) -> pd.Series:
    return serie.astype("string").str.strip()


def _limpa_categoria(serie: pd.Series, nulos: set[str]) -> pd.Series:
    serie = _texto(serie)
    return serie.mask(serie.isin(nulos))


def _limpa_numero(serie: pd.Series, nulos: set[str]) -> pd.Series:
    serie = _texto(serie).mask(lambda s: s.isin(nulos))
    return pd.to_numeric(serie, errors="coerce")


def _indicador(serie: pd.Series, regra) -> pd.Series:
    saida = pd.Series(pd.NA, index=serie.index, dtype="Int8")
    validos = serie.notna()
    saida.loc[validos] = regra(serie.loc[validos]).astype("int8")
    return saida


def carregar_sinasc(caminhos: list[str | Path]) -> pd.DataFrame:
    partes = []
    for caminho in caminhos:
        base = pd.read_parquet(caminho, columns=list(RENOMEAR))
        base["ano"] = _ano_do_arquivo(caminho)
        partes.append(base)
    return pd.concat(partes, ignore_index=True)


def montar_base_modelagem(base: pd.DataFrame) -> pd.DataFrame:
    base = base.rename(columns=RENOMEAR).copy()

    for coluna in ATRIBUTOS_CATEGORICOS + ["parto"]:
        base[coluna] = _limpa_categoria(base[coluna], NULOS.get(coluna, set()))

    for coluna in ATRIBUTOS_NUMERICOS:
        base[coluna] = _limpa_numero(base[coluna], NULOS.get(coluna, set()))

    base = base[base["parto"].isin(["1", "2"])].copy()
    base[COLUNA_ALVO] = (base["parto"] == "2").astype("int8")
    base = base.drop(columns="parto")

    base["idade_pai_ausente"] = base["idade_pai"].isna().astype("int8")
    base["gravidez_multipla"] = _indicador(base["tipo_gravidez"], lambda s: s.isin(["2", "3"]))
    base["cesarea_previa"] = _indicador(base["cesareas_previas"], lambda s: s > 0)
    base["parto_vaginal_previo"] = _indicador(base["partos_vaginais_previos"], lambda s: s > 0)
    base["perda_fetal_previa"] = _indicador(base["perdas_fetais"], lambda s: s > 0)
    base["inicio_prenatal_tardio"] = _indicador(base["mes_inicio_prenatal"], lambda s: s >= 4)
    base["apresentacao_nao_cefalica"] = _indicador(base["apresentacao_fetal"], lambda s: s.isin(["2", "3"]))
    base["pre_termo"] = _indicador(base["semanas_gestacao"], lambda s: s < 37)

    colunas = ["ano", COLUNA_ALVO] + ATRIBUTOS_MODELO
    return base[colunas].copy()

