# Determinantes do parto cesáreo na Bahia — Resultados e confronto com a literatura

**Projeto:** Aprendizado supervisionado sobre SINASC (Bahia, 2022–2024) · IA III
**Autores:** Antoniel Magalhães e Isnan
**Data:** julho de 2026

---

## 1. Objetivo do relatório

Consolidar (a) **o que o nosso modelo obteve** e (b) **o que a literatura epidemiológica brasileira confirma**, mostrando, para cada achado, o trecho verbatim da fonte que o sustenta e a razão pela qual o mapeamento é válido.

A pergunta de pesquisa é interpretativa, não preditiva: *quais fatores estão associados ao parto cesáreo na Bahia e como isso varia entre municípios?* O objetivo nunca foi predição clínica individual, e sim identificar determinantes **anteparto**.

---

## 2. O que obtivemos

### 2.1 Base e alvo

| Item | Valor |
|---|---|
| Anos | 2022, 2023, 2024 (schema idêntico) |
| Registros reunidos | 504.140 |
| Removidos (PARTO ignorado/vazio) | 355 (0,07%) |
| Base para modelagem | **503.785** |
| Balanceamento | ≈ 50,1% cesárea · 49,9% vaginal |
| Tendência da cesárea | 47,9% (2022) → 50,0% (2023) → 52,6% (2024) |

### 2.2 Modelo final (v8)

- **Classificador:** Random Forest (n_estimators=250, max_depth=30, min_samples_split=20, min_samples_leaf=2, max_features=0.3)
- **Validação:** partição hold-out estratificada 80/20 (≈403 mil treino / 101 mil teste)
- **Variáveis:** 18 originais → 53 atributos após codificação
- **Métricas no teste:** AUC-ROC **0,879** · F1 **0,791** · Acurácia **0,793**

### 2.3 Evolução do desempenho (marcos)

| Etapa | Bloco adicionado | AUC | Δ |
|---|---|---|---|
| v0 | idade, cesáreas ant., pré-natal, município | 0,694 | — |
| v2 | + obstétrico e social (paridade, escolaridade, raça/cor) | 0,741 | +0,047 |
| v4 | + gestação e Kotelchuck | 0,748 | +0,007 |
| v5 | + assistência ao parto (inclui TPNASCASSI) | 0,849 | +0,101 |
| v6 | + território (município/estabelecimento) | 0,878 | +0,029 |
| v6 (target encoding) | mesma info, codificação eficiente | 0,883 | +0,006 |
| v7 | + grupo de Robson | 0,900 | +0,017 |
| v7 tuned | RandomizedSearchCV | 0,920 | +0,020 |
| **— sem TPNASCASSI** | remove variável de processo | 0,873 | **−0,047** |
| **v8 (final)** | + 5 determinantes anteparto | **0,879** | +0,006 |

> **Decisão metodológica central:** o maior salto isolado de AUC (+0,101, no v5) veio do `TPNASCASSI` (profissional que assistiu o parto), variável de **processo** registrada no ato do parto. Ela foi **excluída de propósito** por ser incompatível com a pergunta sobre determinantes anteparto — trocando ~5 pontos percentuais de AUC por validade interpretativa.

### 2.4 Importância relativa das variáveis (modelo v8)

Confirmado em `experiments/experiments.jsonl` (exp_019):

| # | Determinante | Importância |
|---|---|---|
| 1 | Taxa de cesárea por grupo Robson | 25,7% |
| 2 | Taxa de cesárea do estabelecimento | 22,5% |
| 3 | Cesáreas anteriores | 10,4% |
| 4 | Taxa de cesárea do município | 7,2% |
| 5 | Idade materna | 7,2% |
| 6 | Partos vaginais anteriores | 6,6% |
| 7 | Idade gestacional | 4,2% |
| 8 | Escolaridade materna | 3,5% |
| — | Kotelchuck 2,5% · estado civil 2,0% · mesmo-município 1,5% · raça/cor RN 1,0% | |

> Os cinco primeiros pesos foram reconferidos diretamente no log; os demais provêm do artefato completo do modelo (`.joblib`).

### 2.5 Codificação (regra de ouro: sem vazamento)

- **Numéricas:** idade, cesáreas/partos vaginais anteriores, semanas de gestação, paridade, mesmo-município.
- **Ordinal:** escolaridade agregada (0–8).
- **One-hot:** tipo de gravidez, raça/cor (mãe e RN), estado civil, sexo, Kotelchuck, apresentação fetal, local de nascimento.
- **Target encoding** (taxa de cesárea suavizada do treino, smoothing = 10): município, estabelecimento e Robson. Reduziu 1.435 → 53 colunas, manteve o AUC e tornou o efeito territorial/clínico interpretável.

---

## 3. Confronto com a literatura

Cada achado abaixo traz o trecho **verbatim** da fonte (original + tradução) e a justificativa do mapeamento.

### 3.1 Robson como eixo clínico dominante
**Nosso achado:** taxa de Robson = variável nº 1 (25,7%).

> **Rocha et al. 2023:** *"Group 5 represents the largest group of live births in this study (over 25%), and has one of the highest CS rates observed (over 85%)."*
> — "O Grupo 5 representa o maior grupo de nascidos vivos deste estudo (mais de 25%), e tem uma das mais altas taxas de cesárea observadas (acima de 85%)."

> **Knobel et al. 2020:** *"Group 5 contributed the most to the overall CS rate, accounting for 30.8% of CSs."*
> — "O Grupo 5 foi o que mais contribuiu para a taxa geral de cesárea, representando 30,8% das cesáreas."

**Por que corrobora:** o estrato de Robson (em especial o G5, definido por cesárea anterior) é o maior discriminador de cesárea no Brasil — coerente com nossa variável de maior importância.

### 3.2 Efeito do estabelecimento de saúde
**Nosso achado:** taxa do estabelecimento = 2ª variável (22,5%); supera o município.

> **Nakamura-Pereira et al. 2016:** *"The overall CS rate was 51.9 %: 42.9 % in the public and 87.9 % in the private sector."*
> — "A taxa geral de cesárea foi de 51,9%: 42,9% no setor público e 87,9% no setor privado."

**Por que corrobora:** um salto de 42,9% → 87,9% conforme a instituição demonstra forte efeito institucional, o que explica o alto poder preditivo da "taxa do estabelecimento" no nosso modelo.

### 3.3 Cesárea anterior
**Nosso achado:** cesáreas anteriores = 3ª variável (10,4%).

> **Shrivastava & Sohn 2025:** *"birth with… a history of CS… was 6.05 times more likely to be via CS"*
> — "um nascimento com… histórico de cesárea… tinha 6,05 vezes mais probabilidade de ser por cesárea."

**Por que corrobora:** OR de 6,05 é o maior efeito individual do estudo. No nosso modelo ela aparece em 3º porque o Robson G5 (nossa nº 1) **já é definido por cesárea anterior**, absorvendo parte do sinal (ver §4).

### 3.4 Idade materna
**Nosso achado:** idade da mãe ≈ 7,2%.

> **Shrivastava & Sohn 2025:** *"the risk of CS increased by 5% for each year increase in the mother's age"*
> — "o risco de cesárea aumentou 5% para cada ano a mais na idade da mãe."

**Por que corrobora:** confirma direção e magnitude moderada — coerente com peso intermediário, não dominante.

### 3.5 Escolaridade materna
**Nosso achado:** escolaridade ≈ 3,5%.

> **Shrivastava & Sohn 2025:** *"compared to women without a high school degree, women with a high school degree or postsecondary education had higher rates of CS births (OR: 1.34 and 2.24, respectively)"*
> — "em comparação com mulheres sem ensino médio completo, mulheres com ensino médio ou ensino superior tiveram taxas mais altas de nascimentos por cesárea (OR: 1,34 e 2,24, respectivamente)."

**Por que corrobora:** confirma o padrão brasileiro (contraintuitivo) de que **mais** escolaridade se associa a **mais** cesárea — determinante social real, de peso menor que o bloco clínico.

### 3.6 Pré-natal (Kotelchuck) — corroboramento mais forte
**Nosso achado:** Kotelchuck pesou pouco (≈2,5%).

> **Piva, Voget & Nucci 2023:** *"No statistically significant associations were found between the adequacy of prenatal care and the rate of cesarean sections in any of the most relevant Robson groups…"* (p = 0,2508)
> — "Não foram encontradas associações estatisticamente significativas entre a adequação do pré-natal e a taxa de cesáreas em nenhum dos grupos de Robson mais relevantes…"

> *"Access to prenatal care… was not associated with the cesarean section rate, suggesting that factors that assess the quality of prenatal care, not simply adequacy of access, should be investigated."*
> — "O acesso ao pré-natal… não esteve associado à taxa de cesárea, sugerindo que fatores que avaliam a *qualidade* do pré-natal, e não simplesmente a adequação do *acesso*, deveriam ser investigados."

**Por que corrobora:** a literatura *confirma ativamente* o efeito nulo — a adequação do pré-natal não discrimina cesárea. Nosso baixo peso é achado alinhado, não ruído.

### 3.7 Raça/cor
**Nosso achado:** raça/cor < 2% (peso pequeno no modelo multivariado).

> **Shrivastava & Sohn 2025:** *"About 52.5% of births to parda women were via CS. Branca and amarela women had higher rates of CS at 66.8% and 58% respectively"*
> — "Cerca de 52,5% dos nascimentos de mulheres pardas foram por cesárea. Mulheres brancas e amarelas tiveram taxas mais altas de cesárea, de 66,8% e 58%, respectivamente."

> *"higher maternal age and educational attainment among branca and amarela significantly mediated their higher CS rates compared to parda women"*
> — "a maior idade materna e a maior escolaridade entre brancas e amarelas *mediaram* significativamente suas taxas mais altas de cesárea em comparação com mulheres pardas."

**Por que corrobora:** existe gradiente racial bruto (66,8% vs 52,5%), mas o artigo mostra que ele é **mediado por idade e escolaridade** — variáveis já presentes no nosso modelo. Assim a raça sobra com pouco peso *condicional*, o que **explica** (em vez de contradizer) nosso < 2%.

---

## 4. Ressalvas metodológicas (para leitura crítica)

1. **"Importância" ≠ "contribuição populacional" ≠ "odds ratio".** Nossa importância é MDI de Random Forest (poder discriminativo). Knobel/Rocha reportam contribuição de grupos ao total de cesáreas; Shrivastava reporta ORs. As métricas não são diretamente comparáveis — o que se confronta é a **direção e a ordem de grandeza**, não o número.

2. **Circularidade parcial do Robson.** O grupo de Robson agrega paridade, apresentação, gestação e cesárea anterior — variáveis também presentes no modelo. Ele deve ser lido como estrato clínico de risco, e explica por que a cesárea anterior, isolada, cai para 3º lugar.

3. **Raça mediada.** O peso baixo da raça reflete confundimento/mediação por escolaridade e território, não ausência de disparidade.

4. **Fonte da Nakamura-Pereira 2016.** Usa o survey "Nascer no Brasil", não o SINASC. O fenômeno (efeito institucional) é o mesmo e bem documentado, mas a fonte difere das demais.

5. **Seleção guiada por teoria.** As variáveis foram escolhidas por plausibilidade epidemiológica + expansão por blocos + ablation, e não por um método automático (LASSO, stepwise, RFE). É uma escolha transparente e coerente com o objetivo interpretativo.

---

## 5. Síntese

O modelo final (AUC 0,879) converge com a literatura brasileira em todos os determinantes principais:

- **Convergência forte:** estabelecimento de saúde, idade materna, escolaridade, adequação do pré-natal (efeito nulo confirmado).
- **Convergência com nuance:** Robson (métrica distinta), cesárea anterior (absorvida pelo Robson), raça/cor (efeito mediado).

Contribuições específicas do trabalho: recorte populacional da Bahia 2022–2024 (≈50% de cesáreas); o **estabelecimento supera o município** na importância relativa; e a exclusão explícita da variável de processo (`TPNASCASSI`) preserva a leitura anteparto.

---

## 6. Referências

1. **Shrivastava S, Sohn H.** Overlapping social structures behind Brazil's cesarean section births: A decomposition analysis. *PLoS One*. 2025;20(6):e0325251. DOI: 10.1371/journal.pone.0325251. https://pmc.ncbi.nlm.nih.gov/articles/PMC12193013/

2. **Rocha AS, Paixão ES, Alves FJO, et al.** Cesarean sections and early-term births according to Robson classification: a population-based study with more than 17 million births in Brazil. *BMC Pregnancy Childbirth*. 2023;23:562. DOI: 10.1186/s12884-023-05807-y. https://pmc.ncbi.nlm.nih.gov/articles/PMC10399022/

3. **Knobel R, Lopes TJP, Menezes MDO, Andreucci CB, Gieburowski JT, Takemoto MLS.** Cesarean-section rates in Brazil from 2014 to 2016: Cross-sectional analysis using the Robson classification. *Rev Bras Ginecol Obstet*. 2020;42(9):522–528. DOI: 10.1055/s-0040-1712134. https://pmc.ncbi.nlm.nih.gov/articles/PMC10309242/

4. **Nakamura-Pereira M, Leal MC, Esteves-Pereira AP, et al.** Use of Robson classification to assess cesarean section rate in Brazil: the role of source of payment for childbirth. *Reproductive Health*. 2016;13(Suppl 3):128. DOI: 10.1186/s12978-016-0228-7. https://pmc.ncbi.nlm.nih.gov/articles/PMC5073850/

5. **Piva VMR, Voget V, Nucci LB.** Cesarean section rates according to the Robson Classification and its association with adequacy levels of prenatal care: a cross-sectional hospital-based study in Brazil. *BMC Pregnancy Childbirth*. 2023;23:455. DOI: 10.1186/s12884-023-05768-2. https://pmc.ncbi.nlm.nih.gov/articles/PMC10283223/
