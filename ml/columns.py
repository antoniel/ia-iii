from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum


class Col(StrEnum):
    """Colunas snake_case (parquet features / train)."""

    CODIGO_ANOMALIA_CONGENITA = "codigo_anomalia_congenita"
    CODIGO_ESTABELECIMENTO_SAUDE = "codigo_estabelecimento_saude"
    CODIGO_MUNICIPIO_NASCIMENTO = "codigo_municipio_nascimento"
    CODIGO_MUNICIPIO_NATURALIDADE_MAE = "codigo_municipio_naturalidade_mae"
    CODIGO_MUNICIPIO_RESIDENCIA = "codigo_municipio_residencia"
    CODIGO_OCUPACAO_MAE = "codigo_ocupacao_mae"
    CODIGO_PAIS_RESIDENCIA = "codigo_pais_residencia"
    CODIGO_UF_NATURALIDADE_MAE = "codigo_uf_naturalidade_mae"
    NUMERO_CONSULTAS_PRENATAL = "numero_consultas_prenatal"
    NUMERO_CONSULTAS_PRENATAL_AGRUPADO = "numero_consultas_prenatal_agrupado"
    ESCOLARIDADE_MAE_ANOS = "escolaridade_mae_anos"
    ESCOLARIDADE_MAE_2010 = "escolaridade_mae_2010"
    ESCOLARIDADE_MAE_AGREGADA = "escolaridade_mae_agregada"
    ESTADO_CIVIL_MAE = "estado_civil_mae"
    SEMANAS_GESTACAO_AGRUPADO = "semanas_gestacao_agrupado"
    TIPO_GRAVIDEZ = "tipo_gravidez"
    IDADE_MAE = "idade_mae"
    IDADE_PAI = "idade_pai"
    ANOMALIA_CONGENITA_IDENTIFICADA = "anomalia_congenita_identificada"
    INDICE_KOTELCHUCK_PRENATAL = "indice_kotelchuck_prenatal"
    LOCAL_NASCIMENTO = "local_nascimento"
    MES_INICIO_PRENATAL = "mes_inicio_prenatal"
    NATURALIDADE_MAE = "naturalidade_mae"
    PARIDADE = "paridade"
    TIPO_PARTO = "tipo_parto"
    QUANTIDADE_PERDAS_FETAIS_ABORTOS = "quantidade_perdas_fetais_abortos"
    QUANTIDADE_FILHOS_VIVOS = "quantidade_filhos_vivos"
    QUANTIDADE_GESTACOES_ANTERIORES = "quantidade_gestacoes_anteriores"
    QUANTIDADE_PARTOS_CESAREOS_ANTERIORES = "quantidade_partos_cesareos_anteriores"
    QUANTIDADE_PARTOS_VAGINAIS_ANTERIORES = "quantidade_partos_vaginais_anteriores"
    RACA_COR_RECEM_NASCIDO = "raca_cor_recem_nascido"
    RACA_COR_MAE = "raca_cor_mae"
    SEMANAS_GESTACAO = "semanas_gestacao"
    SERIE_ESCOLAR_MAE = "serie_escolar_mae"
    SEXO_RECEM_NASCIDO = "sexo_recem_nascido"
    TIPO_APRESENTACAO_FETAL = "tipo_apresentacao_fetal"
    TIPO_DOCUMENTO_RESPONSAVEL_PREENCHIMENTO = (
        "tipo_documento_responsavel_preenchimento"
    )
    TIPO_FUNCAO_RESPONSAVEL_PREENCHIMENTO = "tipo_funcao_responsavel_preenchimento"
    TIPO_METODO_ESTIMATIVA_GESTACIONAL = "tipo_metodo_estimativa_gestacional"
    TIPO_PROFISSIONAL_ASSISTENCIA_PARTO = "tipo_profissional_assistencia_parto"
    GRUPO_ROBSON = "grupo_robson"
    NASCIMENTO_MESMO_MUNICIPIO_RESIDENCIA = "nascimento_mesmo_municipio_residencia"
    ANO = "ano"
    Y_CESAREA = "y_cesarea"


class Sinasc(StrEnum):
    """Colunas originais SINASC (parquet preprocess)."""

    CODANOMAL = "CODANOMAL"
    CODESTAB = "CODESTAB"
    CODMUNNASC = "CODMUNNASC"
    CODMUNNATU = "CODMUNNATU"
    CODMUNRES = "CODMUNRES"
    CODOCUPMAE = "CODOCUPMAE"
    CODPAISRES = "CODPAISRES"
    CODUFNATU = "CODUFNATU"
    CONSPRENAT = "CONSPRENAT"
    CONSULTAS = "CONSULTAS"
    ESCMAE = "ESCMAE"
    ESCMAE2010 = "ESCMAE2010"
    ESCMAEAGR1 = "ESCMAEAGR1"
    ESTCIVMAE = "ESTCIVMAE"
    GESTACAO = "GESTACAO"
    GRAVIDEZ = "GRAVIDEZ"
    IDADEMAE = "IDADEMAE"
    IDADEPAI = "IDADEPAI"
    IDANOMAL = "IDANOMAL"
    KOTELCHUCK = "KOTELCHUCK"
    LOCNASC = "LOCNASC"
    MESPRENAT = "MESPRENAT"
    NATURALMAE = "NATURALMAE"
    PARIDADE = "PARIDADE"
    PARTO = "PARTO"
    QTDFILMORT = "QTDFILMORT"
    QTDFILVIVO = "QTDFILVIVO"
    QTDGESTANT = "QTDGESTANT"
    QTDPARTCES = "QTDPARTCES"
    QTDPARTNOR = "QTDPARTNOR"
    RACACOR = "RACACOR"
    RACACORMAE = "RACACORMAE"
    SEMAGESTAC = "SEMAGESTAC"
    SERIESCMAE = "SERIESCMAE"
    SEXO = "SEXO"
    TPAPRESENT = "TPAPRESENT"
    TPDOCRESP = "TPDOCRESP"
    TPFUNCRESP = "TPFUNCRESP"
    TPMETESTIM = "TPMETESTIM"
    TPNASCASSI = "TPNASCASSI"
    TPROBSON = "TPROBSON"


SINASC_TO_COL: dict[Sinasc, Col] = {
    Sinasc.CODANOMAL: Col.CODIGO_ANOMALIA_CONGENITA,
    Sinasc.CODESTAB: Col.CODIGO_ESTABELECIMENTO_SAUDE,
    Sinasc.CODMUNNASC: Col.CODIGO_MUNICIPIO_NASCIMENTO,
    Sinasc.CODMUNNATU: Col.CODIGO_MUNICIPIO_NATURALIDADE_MAE,
    Sinasc.CODMUNRES: Col.CODIGO_MUNICIPIO_RESIDENCIA,
    Sinasc.CODOCUPMAE: Col.CODIGO_OCUPACAO_MAE,
    Sinasc.CODPAISRES: Col.CODIGO_PAIS_RESIDENCIA,
    Sinasc.CODUFNATU: Col.CODIGO_UF_NATURALIDADE_MAE,
    Sinasc.CONSPRENAT: Col.NUMERO_CONSULTAS_PRENATAL,
    Sinasc.CONSULTAS: Col.NUMERO_CONSULTAS_PRENATAL_AGRUPADO,
    Sinasc.ESCMAE: Col.ESCOLARIDADE_MAE_ANOS,
    Sinasc.ESCMAE2010: Col.ESCOLARIDADE_MAE_2010,
    Sinasc.ESCMAEAGR1: Col.ESCOLARIDADE_MAE_AGREGADA,
    Sinasc.ESTCIVMAE: Col.ESTADO_CIVIL_MAE,
    Sinasc.GESTACAO: Col.SEMANAS_GESTACAO_AGRUPADO,
    Sinasc.GRAVIDEZ: Col.TIPO_GRAVIDEZ,
    Sinasc.IDADEMAE: Col.IDADE_MAE,
    Sinasc.IDADEPAI: Col.IDADE_PAI,
    Sinasc.IDANOMAL: Col.ANOMALIA_CONGENITA_IDENTIFICADA,
    Sinasc.KOTELCHUCK: Col.INDICE_KOTELCHUCK_PRENATAL,
    Sinasc.LOCNASC: Col.LOCAL_NASCIMENTO,
    Sinasc.MESPRENAT: Col.MES_INICIO_PRENATAL,
    Sinasc.NATURALMAE: Col.NATURALIDADE_MAE,
    Sinasc.PARIDADE: Col.PARIDADE,
    Sinasc.PARTO: Col.TIPO_PARTO,
    Sinasc.QTDFILMORT: Col.QUANTIDADE_PERDAS_FETAIS_ABORTOS,
    Sinasc.QTDFILVIVO: Col.QUANTIDADE_FILHOS_VIVOS,
    Sinasc.QTDGESTANT: Col.QUANTIDADE_GESTACOES_ANTERIORES,
    Sinasc.QTDPARTCES: Col.QUANTIDADE_PARTOS_CESAREOS_ANTERIORES,
    Sinasc.QTDPARTNOR: Col.QUANTIDADE_PARTOS_VAGINAIS_ANTERIORES,
    Sinasc.RACACOR: Col.RACA_COR_RECEM_NASCIDO,
    Sinasc.RACACORMAE: Col.RACA_COR_MAE,
    Sinasc.SEMAGESTAC: Col.SEMANAS_GESTACAO,
    Sinasc.SERIESCMAE: Col.SERIE_ESCOLAR_MAE,
    Sinasc.SEXO: Col.SEXO_RECEM_NASCIDO,
    Sinasc.TPAPRESENT: Col.TIPO_APRESENTACAO_FETAL,
    Sinasc.TPDOCRESP: Col.TIPO_DOCUMENTO_RESPONSAVEL_PREENCHIMENTO,
    Sinasc.TPFUNCRESP: Col.TIPO_FUNCAO_RESPONSAVEL_PREENCHIMENTO,
    Sinasc.TPMETESTIM: Col.TIPO_METODO_ESTIMATIVA_GESTACIONAL,
    Sinasc.TPNASCASSI: Col.TIPO_PROFISSIONAL_ASSISTENCIA_PARTO,
    Sinasc.TPROBSON: Col.GRUPO_ROBSON,
}

COLUMN_RENAMES: dict[str, str] = {k.value: v.value for k, v in SINASC_TO_COL.items()}

TARGET = Col.Y_CESAREA

FEATURES_V0: tuple[Col, ...] = (
    Col.IDADE_MAE,
    Col.QUANTIDADE_PARTOS_CESAREOS_ANTERIORES,
    Col.NUMERO_CONSULTAS_PRENATAL,
    Col.CODIGO_MUNICIPIO_RESIDENCIA,
)

FEATURES_V1: tuple[Col, ...] = FEATURES_V0 + (
    Col.PARIDADE,
    Col.TIPO_GRAVIDEZ,
)

FEATURES_V2: tuple[Col, ...] = FEATURES_V1 + (
    Col.ESCOLARIDADE_MAE_AGREGADA,
    Col.RACA_COR_MAE,
)

FEATURES_V3: tuple[Col, ...] = FEATURES_V2 + (Col.SEMANAS_GESTACAO,)

FEATURES_V4: tuple[Col, ...] = tuple(
    col
    for col in FEATURES_V3
    if col is not Col.NUMERO_CONSULTAS_PRENATAL
) + (Col.INDICE_KOTELCHUCK_PRENATAL,)

FEATURES_V5: tuple[Col, ...] = FEATURES_V4 + (
    Col.TIPO_APRESENTACAO_FETAL,
    Col.TIPO_PROFISSIONAL_ASSISTENCIA_PARTO,
)

FEATURES_V6: tuple[Col, ...] = FEATURES_V5 + (
    Col.LOCAL_NASCIMENTO,
    Col.CODIGO_ESTABELECIMENTO_SAUDE,
)

FEATURES_V6_SEM_MUNICIPIO: tuple[Col, ...] = tuple(
    col for col in FEATURES_V6 if col is not Col.CODIGO_MUNICIPIO_RESIDENCIA
)

FEATURES_V7: tuple[Col, ...] = FEATURES_V6 + (Col.GRUPO_ROBSON,)

FEATURES_V7_SEM_TPNASCASSI: tuple[Col, ...] = tuple(
    col for col in FEATURES_V7 if col is not Col.TIPO_PROFISSIONAL_ASSISTENCIA_PARTO
)

FEATURES_V8: tuple[Col, ...] = FEATURES_V7_SEM_TPNASCASSI + (
    Col.QUANTIDADE_PARTOS_VAGINAIS_ANTERIORES,
    Col.NASCIMENTO_MESMO_MUNICIPIO_RESIDENCIA,
    Col.ESTADO_CIVIL_MAE,
    Col.SEXO_RECEM_NASCIDO,
    Col.RACA_COR_RECEM_NASCIDO,
)

DERIVED_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {Col.NASCIMENTO_MESMO_MUNICIPIO_RESIDENCIA.value}
)

# ESCMAEAGR1: escada 00–08 (dicionário SINASC); 09 ignorado; 10–12 ≈ incompleto no nível.
ESCOLARIDADE_MAE_AGREGADA_ORDINAL: dict[str, int | None] = {
    "00": 0,
    "01": 1,
    "02": 2,
    "03": 3,
    "04": 4,
    "05": 5,
    "06": 6,
    "07": 7,
    "08": 8,
    "09": None,
    "10": 1,
    "11": 3,
    "12": 5,
    "": None,
}

ORDINAL_FEATURES: frozenset[Col] = frozenset({Col.ESCOLARIDADE_MAE_AGREGADA})

ORDINAL_MAPS: dict[Col, dict[str, int | None]] = {
    Col.ESCOLARIDADE_MAE_AGREGADA: ESCOLARIDADE_MAE_AGREGADA_ORDINAL,
}

CATEGORICAL_FEATURES: frozenset[Col] = frozenset(
    {
        Col.CODIGO_MUNICIPIO_RESIDENCIA,
        Col.TIPO_GRAVIDEZ,
        Col.RACA_COR_MAE,
        Col.RACA_COR_RECEM_NASCIDO,
        Col.ESTADO_CIVIL_MAE,
        Col.SEXO_RECEM_NASCIDO,
        Col.INDICE_KOTELCHUCK_PRENATAL,
        Col.TIPO_APRESENTACAO_FETAL,
        Col.TIPO_PROFISSIONAL_ASSISTENCIA_PARTO,
        Col.LOCAL_NASCIMENTO,
        Col.CODIGO_ESTABELECIMENTO_SAUDE,
        Col.GRUPO_ROBSON,
    }
)

# Target encoding: taxa de cesárea suavizada do treino (1 coluna por feature).
TARGET_ENCODE_FEATURES: frozenset[Col] = frozenset(
    {
        Col.CODIGO_MUNICIPIO_RESIDENCIA,
        Col.CODIGO_ESTABELECIMENTO_SAUDE,
        Col.GRUPO_ROBSON,
    }
)


def target_encode_column_name(col: Col) -> str:
    return f"{col.value}_taxa_cesarea"


def col_names(columns: Sequence[Col]) -> tuple[str, ...]:
    return tuple(c.value for c in columns)
