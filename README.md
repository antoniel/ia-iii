# IA-III

Trabalho de aprendizado supervisionado com dados abertos do SUS (SINASC).

## Setup

```bash
uv sync
```

## Sequencia de execucao

Gerar a base de modelagem:

```bash
python scripts/gera_base_modelagem.py
```

Treinar os modelos e salvar os artefatos:

```bash
python scripts/treina_modelos.py
```

Abrir o caderno para leitura dos resultados:

```bash
uv run jupyter lab notebooks/
```

## O que cada script faz

- `scripts/gera_base_modelagem.py`: junta os arquivos anuais do SINASC, limpa as colunas escolhidas e cria a base final de modelagem.
- `scripts/treina_modelos.py`: separa treino (`2022-2023`) e teste (`2024`), treina regressao logistica e floresta aleatoria, calcula metricas e salva os artefatos.

## Onde ficam os resultados

- Base pronta: `dados/processados/base_modelagem_sinasc.parquet`
- Resultados da modelagem: `dados/processados/resultados_modelagem/`

Arquivos principais gerados:

- `metricas_modelos.csv`
- `matriz_confusao_regressao_logistica.csv`
- `matriz_confusao_floresta_aleatoria.csv`
- `coeficientes_regressao_logistica.csv`
- `importancias_floresta.csv`
- `importancias_permutacao_floresta.csv`

## Estrutura

- `ml/`: engenharia de atributos e modelagem
- `notebooks/`: leitura e visualizacao dos resultados
- `scripts/`: geracao da base e treino
- `dados/`: dados locais e artefatos
- `slides/`: apresentacao
