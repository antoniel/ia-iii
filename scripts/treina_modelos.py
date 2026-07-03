from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from ml import (  # noqa: E402
    ATRIBUTOS_CATEGORICOS,
    ATRIBUTOS_DERIVADOS,
    ATRIBUTOS_NUMERICOS,
    calcular_metricas,
    carregar_base_modelagem,
    coeficientes_logistica,
    criar_pipeline_floresta,
    criar_pipeline_logistica,
    dividir_treino_teste,
    importancias_floresta,
    importancias_permutacao,
    matriz_confusao_tabela,
    separar_xy_ano,
)

PASTA_SAIDA = Path("dados/processados/resultados_modelagem")


def _salvar_json(caminho: Path, dados: dict[str, object]) -> None:
    caminho.write_text(json.dumps(dados, indent=2), encoding="utf-8")


def _treinar_modelo(nome: str, pipeline, x_treino, y_treino, x_teste, y_teste) -> dict[str, object]:
    pipeline.fit(x_treino, y_treino)
    y_prev = pipeline.predict(x_teste)
    y_prob = pipeline.predict_proba(x_teste)[:, 1]
    return {
        "modelo": nome,
        "metricas": calcular_metricas(y_teste, y_prev, y_prob),
        "matriz": matriz_confusao_tabela(y_teste, y_prev),
        "pipeline": pipeline,
    }


def main() -> None:
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    base = carregar_base_modelagem()
    x, y, ano = separar_xy_ano(base)
    partes = dividir_treino_teste(x, y, ano)

    x_treino = partes["x_treino"]
    x_teste = partes["x_teste"]
    y_treino = partes["y_treino"]
    y_teste = partes["y_teste"]

    print("Etapa 1 - preparo para treino")
    print(f"Atributos em X: {x.shape[1]}")
    print("Ano nao entra em X; ele so define o corte temporal.")
    print("Ausentes numericos usam mediana; categoricos usam a moda.")
    print("Categorias sao codificadas porque os modelos precisam de entrada numerica.")
    print()

    resultado_logistica = _treinar_modelo(
        "regressao_logistica",
        criar_pipeline_logistica(),
        x_treino,
        y_treino,
        x_teste,
        y_teste,
    )
    print("Etapa 2 - baseline")
    print("O baseline e a regressao logistica.")
    print("Ela usa o mesmo corte temporal e as mesmas variaveis do modelo principal.")
    print("Coeficiente positivo aumenta a chance prevista de cesarea; negativo reduz.")
    print()

    resultado_floresta = _treinar_modelo(
        "floresta_aleatoria",
        criar_pipeline_floresta(),
        x_treino,
        y_treino,
        x_teste,
        y_teste,
    )
    print("Etapa 3 - modelo principal")
    print("A floresta foi escolhida por capturar interacoes e relacoes nao lineares.")
    print("Importancia por arvore resume o uso interno das variaveis.")
    print("Importancia por permutacao mede a perda de F1 quando um atributo e embaralhado.")
    print()

    metricas = []
    for resultado in [resultado_logistica, resultado_floresta]:
        nome = resultado["modelo"]
        dados_metricas = {"modelo": nome, **resultado["metricas"]}
        metricas.append(dados_metricas)
        _salvar_json(PASTA_SAIDA / f"metricas_{nome}.json", dados_metricas)
        resultado["matriz"].to_csv(PASTA_SAIDA / f"matriz_confusao_{nome}.csv")

    pd.DataFrame(metricas).to_csv(PASTA_SAIDA / "metricas_modelos.csv", index=False)
    coeficientes_logistica(resultado_logistica["pipeline"]).to_csv(
        PASTA_SAIDA / "coeficientes_regressao_logistica.csv",
        index=False,
    )
    importancias_floresta(resultado_floresta["pipeline"]).to_csv(
        PASTA_SAIDA / "importancias_floresta.csv",
        index=False,
    )
    importancias_permutacao(resultado_floresta["pipeline"], x_teste, y_teste).to_csv(
        PASTA_SAIDA / "importancias_permutacao_floresta.csv",
        index=False,
    )

    resumo_divisao = {
        "linhas_treino": int(len(x_treino)),
        "linhas_teste": int(len(x_teste)),
        "anos_treino": sorted(partes["ano_treino"].unique().tolist()),
        "anos_teste": sorted(partes["ano_teste"].unique().tolist()),
        "atributos_numericos": ATRIBUTOS_NUMERICOS + ATRIBUTOS_DERIVADOS,
        "atributos_categoricos": ATRIBUTOS_CATEGORICOS,
    }
    _salvar_json(PASTA_SAIDA / "resumo_divisao.json", resumo_divisao)

    melhor = pd.DataFrame(metricas).sort_values("f1_cesarea", ascending=False).iloc[0]["modelo"]
    print("Etapa 4 - comparacao e artefatos")
    print(f"Melhor modelo por F1 de cesarea: {melhor}")
    print(f"Arquivos salvos em: {PASTA_SAIDA}")


if __name__ == "__main__":
    main()

