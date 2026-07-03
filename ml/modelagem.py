from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .engenharia_atributos import (
    ATRIBUTOS_CATEGORICOS,
    ATRIBUTOS_DERIVADOS,
    ATRIBUTOS_MODELO,
    ATRIBUTOS_NUMERICOS,
    COLUNA_ALVO,
)

CAMINHO_BASE = Path("dados/processados/base_modelagem_sinasc.parquet")
COLUNAS_NUMERICAS_MODELO = ATRIBUTOS_NUMERICOS + ATRIBUTOS_DERIVADOS
ANOS_TREINO = (2022, 2023)
ANO_TESTE = 2024


def carregar_base_modelagem(caminho: str | Path = CAMINHO_BASE) -> pd.DataFrame:
    return pd.read_parquet(caminho)


def separar_xy_ano(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = base[ATRIBUTOS_MODELO].copy()
    y = base[COLUNA_ALVO].copy()
    ano = base["ano"].copy()
    return x, y, ano


def dividir_treino_teste(
    x: pd.DataFrame,
    y: pd.Series,
    ano: pd.Series,
    anos_treino: tuple[int, ...] = ANOS_TREINO,
    ano_teste: int = ANO_TESTE,
) -> dict[str, pd.DataFrame | pd.Series]:
    mascara_treino = ano.isin(anos_treino)
    mascara_teste = ano.eq(ano_teste)

    if not mascara_treino.any():
        raise ValueError("Nenhuma linha encontrada para treino.")
    if not mascara_teste.any():
        raise ValueError("Nenhuma linha encontrada para teste.")

    return {
        "x_treino": x.loc[mascara_treino].copy(),
        "x_teste": x.loc[mascara_teste].copy(),
        "y_treino": y.loc[mascara_treino].copy(),
        "y_teste": y.loc[mascara_teste].copy(),
        "ano_treino": ano.loc[mascara_treino].copy(),
        "ano_teste": ano.loc[mascara_teste].copy(),
    }


def montar_preprocessador(
    colunas_numericas: list[str] | None = None,
    colunas_categoricas: list[str] | None = None,
    padronizar: bool = False,
) -> ColumnTransformer:
    colunas_numericas = colunas_numericas or COLUNAS_NUMERICAS_MODELO
    colunas_categoricas = colunas_categoricas or ATRIBUTOS_CATEGORICOS

    passos_numericos: list[tuple[str, object]] = [
        ("imputador", SimpleImputer(strategy="median", missing_values=pd.NA))
    ]
    if padronizar:
        passos_numericos.append(("padronizador", StandardScaler()))

    fluxo_numerico = Pipeline(passos_numericos)
    fluxo_categorico = Pipeline(
        [
            ("imputador", SimpleImputer(strategy="most_frequent", missing_values=pd.NA)),
            ("codificador", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("numericas", fluxo_numerico, colunas_numericas),
            ("categoricas", fluxo_categorico, colunas_categoricas),
        ],
        verbose_feature_names_out=False,
    )


def criar_pipeline_logistica() -> Pipeline:
    return Pipeline(
        [
            ("preparo", montar_preprocessador(padronizar=True)),
            ("modelo", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def criar_pipeline_floresta() -> Pipeline:
    return Pipeline(
        [
            ("preparo", montar_preprocessador(padronizar=False)),
            (
                "modelo",
                RandomForestClassifier(
                    n_estimators=120,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=42,
                ),
            ),
        ]
    )


def calcular_metricas(
    y_real: pd.Series,
    y_prev: pd.Series,
    y_prob: pd.Series,
) -> dict[str, float]:
    return {
        "acuracia": float(accuracy_score(y_real, y_prev)),
        "precisao_cesarea": float(precision_score(y_real, y_prev, pos_label=1)),
        "recall_cesarea": float(recall_score(y_real, y_prev, pos_label=1)),
        "f1_cesarea": float(f1_score(y_real, y_prev, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_real, y_prob)),
    }


def matriz_confusao_tabela(y_real: pd.Series, y_prev: pd.Series) -> pd.DataFrame:
    matriz = confusion_matrix(y_real, y_prev, labels=[0, 1])
    return pd.DataFrame(
        matriz,
        index=["real_vaginal", "real_cesarea"],
        columns=["prev_vaginal", "prev_cesarea"],
    )


def nomes_atributos_expandidos(pipeline: Pipeline) -> list[str]:
    return pipeline.named_steps["preparo"].get_feature_names_out().tolist()


def coeficientes_logistica(pipeline: Pipeline) -> pd.DataFrame:
    tabela = pd.DataFrame(
        {
            "atributo": nomes_atributos_expandidos(pipeline),
            "coeficiente": pipeline.named_steps["modelo"].coef_[0],
        }
    )
    return tabela.reindex(tabela["coeficiente"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def importancias_floresta(pipeline: Pipeline) -> pd.DataFrame:
    tabela = pd.DataFrame(
        {
            "atributo": nomes_atributos_expandidos(pipeline),
            "importancia": pipeline.named_steps["modelo"].feature_importances_,
        }
    )
    return tabela.sort_values("importancia", ascending=False).reset_index(drop=True)


def importancias_permutacao(
    pipeline: Pipeline,
    x_teste: pd.DataFrame,
    y_teste: pd.Series,
) -> pd.DataFrame:
    resultado = permutation_importance(
        pipeline,
        x_teste,
        y_teste,
        n_repeats=3,
        random_state=42,
        scoring=make_scorer(f1_score, pos_label=1),
        n_jobs=1,
    )
    tabela = pd.DataFrame(
        {
            "atributo": x_teste.columns.tolist(),
            "media": resultado.importances_mean,
            "desvio": resultado.importances_std,
        }
    )
    return tabela.sort_values("media", ascending=False).reset_index(drop=True)
