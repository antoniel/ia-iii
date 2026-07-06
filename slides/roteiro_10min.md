# Roteiro de apresentação: 10 minutos

## Objetivo da fala

Defender que o trabalho não é uma ferramenta de predição clínica individual. O modelo é usado como instrumento para identificar fatores associados ao parto cesáreo na Bahia e entender como esses fatores aparecem no território.

Tempo alvo: 10 minutos.

## Distribuição de tempo

- Título e agenda: 30 s
- Problema e objetivos: 1 min 20 s
- Dados e limpeza: 1 min 40 s
- Variáveis e metodologia: 2 min 40 s
- Resultados: 2 min 40 s
- Síntese e conclusão: 1 min 10 s

## Slide 1: Título, 15 s

Bom dia. Nós somos Antoniel e Isnan. Neste trabalho, analisamos os determinantes associados ao parto cesáreo na Bahia usando dados abertos do SINASC, que é o Sistema de Informações sobre Nascidos Vivos do DATASUS.

A ideia central não é vender um classificador de cesárea. A ideia é usar aprendizado supervisionado como uma lupa para entender quais fatores aparecem associados ao tipo de parto.

## Slide 2: Agenda, 15 s

A apresentação tem duas partes principais. Primeiro, vamos mostrar como construímos o modelo: dados, limpeza, escolha de variáveis e validação. Depois, vamos mostrar os resultados: desempenho, importância das variáveis e como esses achados dialogam com a literatura.

## Slide 3: Problema, 50 s

A pergunta do trabalho é: quais fatores estão associados ao parto cesáreo na Bahia e como isso varia entre municípios?

A base usada foi o SINASC, com declarações de nascidos vivos na Bahia de 2022 a 2024. O alvo é o tipo de parto, separando parto vaginal e cesárea.

Um ponto importante é o limite da interpretação. Como o SINASC é um registro retrospectivo, quando a informação está na base o parto já aconteceu. Então não faria sentido apresentar isso como uma ferramenta de decisão clínica individual. O foco é epidemiológico: identificar padrões, determinantes associados e diferenças territoriais.

## Slide 4: Objetivos, 30 s

A partir dessa pergunta, os objetivos foram quatro. Primeiro, desenvolver e validar um modelo supervisionado. Segundo, identificar a importância relativa dos determinantes. Terceiro, descrever a variação entre municípios. E quarto, comparar os achados com a literatura sobre cesárea, saúde materna e SINASC.

Então a métrica do modelo é importante, mas ela serve como suporte metodológico. O produto principal é a interpretação dos fatores associados.

## Slide 5: Metodologia, 45 s

A metodologia foi construída de forma iterativa. Nós começamos com poucas variáveis, todas plausíveis do ponto de vista epidemiológico, e fomos expandindo por blocos: obstétrico, social, gestacional, assistência, território e estratificação clínica.

A cada bloco, avaliamos se havia ganho discriminativo e se a variável fazia sentido para a pergunta. Também excluímos variáveis que indicavam processo assistencial ou informação pós-parto, porque elas poderiam aumentar a AUC, mas piorar a interpretação como determinante anteparto.

## Slide 6: Pré-processamento, fontes, 35 s

A base reuniu três anos de declarações de nascidos vivos na Bahia. Em 2022 foram cerca de 174 mil registros, em 2023 cerca de 170 mil, e em 2024 cerca de 160 mil. No total, começamos com 504.140 registros.

Essa escala é uma vantagem do SINASC: mesmo depois das limpezas, a base continua populacional e grande o suficiente para análise por subgrupos e território.

## Slide 7: Pré-processamento, rótulo do parto, 45 s

A primeira limpeza foi no próprio alvo. O campo PARTO tem os códigos 1 para vaginal, 2 para cesárea, 9 para ignorado, além de valores em branco.

Mantivemos apenas as classes válidas, vaginal e cesárea, e removemos ignorados e vazios. Isso retirou só 355 registros, cerca de 0,07 por cento da base. Ou seja, a qualidade do campo principal é muito boa para esse recorte.

## Slide 8: Base unificada, 30 s

Depois da unificação dos anos e da limpeza do rótulo, ficamos com 503.785 registros utilizáveis.

Outro ponto bom para modelagem é que o desfecho ficou praticamente balanceado: aproximadamente 50,1 por cento cesárea e 49,9 por cento parto vaginal. Então não estamos lidando com um problema de classe rara.

## Slide 9: O que excluímos, 45 s

Antes de treinar o modelo, retiramos campos que poderiam gerar vazamento ou que não respondiam à pergunta.

Entram nesse grupo identificadores administrativos, datas auxiliares, peso e Apgar ao nascer, além de variáveis diretamente ligadas ao processo do parto, como indução e momento da cesárea.

O critério foi: manter apenas o que poderia ser conhecido antes do desfecho obstétrico ou que representasse contexto anterior ao parto. Isso é essencial para não confundir boa classificação com vazamento de informação.

## Slide 10: Variáveis iniciais, 40 s

O modelo inicial começou com quatro variáveis: idade da mãe, cesáreas anteriores, consultas de pré-natal e município de residência.

Essas variáveis foram escolhidas por plausibilidade. Idade materna avançada pode estar associada a maior risco obstétrico. Cesárea anterior é um histórico muito relevante. Pré-natal indica cobertura assistencial. E município captura contexto territorial.

Com esse conjunto inicial, a AUC foi 0,694. Isso serviu como referência para as próximas etapas.

## Slide 11: Expansão do conjunto de variáveis, 1 min

Depois, expandimos o conjunto por blocos temáticos. Primeiro entraram variáveis obstétricas, como paridade e tipo de gravidez. Depois entraram características sociais da mãe, como escolaridade e raça/cor. Em seguida, variáveis de gestação e pré-natal, como idade gestacional e Kotelchuck.

Também testamos variáveis de assistência ao parto, territorialidade, estabelecimento de saúde e classificação de Robson. No modelo final, ficamos com 18 variáveis originais, que após codificação viraram 53 atributos.

O ponto aqui é que a seleção não foi só automática. Cada inclusão precisava ter plausibilidade epidemiológica e não podia representar vazamento pós-parto.

## Slide 12: Fluxo analítico, 1 min

O fluxo foi: dados SINASC, limpeza, criação de variáveis, codificação, divisão treino e teste, Random Forest e métricas.

Na codificação, usamos estratégias diferentes para tipos diferentes de variável. Variáveis nominais de baixa cardinalidade foram codificadas com variáveis indicadoras. Escolaridade materna foi tratada como escala ordinal. E variáveis de alta cardinalidade, como município, estabelecimento e Robson, foram codificadas pela taxa de cesárea suavizada, sempre calculada apenas no treino.

A validação principal foi hold-out estratificado 80/20, com cerca de 403 mil registros no treino e 101 mil no teste. Também rodamos validação cruzada estratificada em alguns experimentos, para checar se as métricas eram estáveis, mas preferimos o hold-out no dia a dia porque, com mais de 500 mil registros, cada rodada demorava bastante e precisávamos iterar rápido na escolha de variáveis.

O Random Forest foi usado porque permite bom desempenho e fornece importâncias de variáveis para análise.

## Slide 13: Alta cardinalidade e Robson, 50 s

Para município, estabelecimento e Robson, usamos uma taxa suavizada de cesárea no conjunto de treino. A suavização evita que categorias com poucos casos tenham taxas extremas demais.

A classificação de Robson é importante porque é recomendada pela OMS e organiza gestantes por características clínicas e obstétricas, como paridade, cesárea anterior, apresentação fetal e idade gestacional.

Mas ela tem uma limitação interpretativa: parte dessas informações já aparece em outras variáveis. Então o Robson ajuda na estratificação, mas não deve ser lido como um fator causal isolado.

## Slide 14: Exclusão do profissional assistente, 55 s

Uma decisão metodológica importante foi excluir a categoria do profissional que assistiu o parto.

Essa variável aumentava bastante o desempenho, com ganho de aproximadamente 5 pontos percentuais de AUC. Mas ela representa o processo assistencial no momento do parto, não um determinante anteparto. Se nosso objetivo fosse só prever o desfecho retrospectivamente, talvez ela entrasse. Mas para a nossa pergunta epidemiológica, ela distorce a interpretação.

Então o modelo final aceita uma AUC um pouco menor em troca de uma leitura mais honesta dos determinantes.

## Slide 15: Evolução do desempenho, 1 min

Antes da tabela, vale fixar como avaliamos o modelo. Separamos 20 por cento da base como conjunto de teste, estratificado por tipo de parto, e treinamos só nos outros 80 por cento. Todas as métricas que reportamos vêm desse teste, que o modelo nunca viu no treino. Em alguns marcos, rodamos também validação cruzada estratificada de 5 folds; a AUC ficou estável, com desvio de cerca de 0,001, o que nos deu confiança de que o hold-out não estava distorcendo o resultado.

Usamos três métricas, sempre como suporte metodológico, não como produto do trabalho. A **AUC-ROC** mede a capacidade de separar cesárea e parto vaginal em diferentes limiares de decisão; quanto mais perto de 1, melhor a separação. O **F1** equilibra acerto em cesáreas e partos vaginais, o que importa numa base quase balanceada. A **acurácia** é a proporção de classificações corretas no total; sozinha ela conta menos, porque cesárea e vaginal estão em proporções parecidas.

A tabela mostra como o desempenho evoluiu ao longo dos experimentos, sempre no mesmo hold-out. O modelo inicial tinha AUC 0,694. Com blocos obstétricos e sociais, subiu para 0,741. Com gestação e pré-natal, chegou a 0,748.

Quando entram variáveis de assistência ao parto, há um salto para 0,849, justamente por causa de variáveis mais próximas do processo. Com territorialidade codificada, o modelo chega a 0,883. Com Robson e profissional assistente, chega a 0,920, mas esse é o modelo de referência com variável de processo.

O modelo final, sem profissional assistente, fica com AUC 0,879, F1 0,791 e acurácia 0,793. Esse foi o compromisso que escolhemos entre desempenho e interpretação.

## Slide 16: Desempenho do modelo final, 35 s

Esses números indicam que o modelo aprendeu padrões consistentes da base. Mas, de novo, não os apresentamos como produto clínico. Eles nos dão confiança para olhar as importâncias relativas das variáveis e discutir se os padrões fazem sentido epidemiológico.

## Slide 17: Importância relativa das variáveis, 1 min 10 s

Nos resultados de importância, os maiores pesos ficaram com três blocos.

O primeiro é o estrato clínico-obstétrico, representado pela taxa de cesárea por grupo de Robson, com cerca de 25,7 por cento da importância. O segundo é o estabelecimento de saúde, com 22,5 por cento. Isso sugere que o contexto institucional tem muito peso. O terceiro é histórico obstétrico, especialmente cesáreas anteriores, com 10,4 por cento.

Também aparecem município de residência, idade materna, partos vaginais anteriores, idade gestacional e escolaridade.

A leitura principal é que cesárea não aparece só como resultado de características individuais da gestante. Há um componente clínico, um componente histórico e um componente institucional e territorial muito fortes.

## Slide 18: Confronto com a literatura, 1 min

Quando comparamos com a literatura, os achados são coerentes.

Cesárea anterior é frequentemente apontada como um dos principais preditores. Idade materna mais alta também aparece associada a maior taxa de cesárea. Escolaridade mais alta costuma aparecer associada a mais cesáreas no Brasil. Robson é um padrão recomendado para interpretar grupos obstétricos. E o estabelecimento de saúde também é descrito na literatura como uma fonte importante de variação.

Então nosso objetivo aqui não é dizer que o modelo descobriu causalidade. Queremos mostrar que os padrões aprendidos pelo modelo são plausíveis e compatíveis com evidências epidemiológicas.

## Slide 19: Síntese dos achados, 1 min

A síntese é que o modelo confirmou alguns determinantes esperados, como histórico obstétrico, idade, escolaridade e estratificação de Robson.

Mas também destacou um ponto importante: o estabelecimento de saúde teve mais importância que o município de residência. Isso sugere que a prática institucional e o local onde o parto acontece podem capturar mais sinal do que apenas o território de residência.

Outra contribuição foi metodológica: quando excluímos o profissional assistente, o ranking muda e a interpretação fica mais alinhada com determinantes anteparto. Então a forma de construir o modelo afeta diretamente a história que contamos.

## Slide 20: Obrigado, 20 s

Para concluir: usamos aprendizado supervisionado como ferramenta de análise. O modelo teve desempenho adequado, mas a contribuição principal foi organizar evidências sobre fatores associados ao parto cesáreo na Bahia e mostrar a importância do contexto clínico, institucional e territorial.

Obrigado. Estamos abertos a perguntas.

## Frases para perguntas prováveis

### Por que Random Forest?

Escolhemos Random Forest porque combina bom desempenho com importância de variáveis, sem assumir relação linear entre os fatores. Como nosso objetivo era explorar determinantes e não estimar odds ratios, ela serviu como modelo principal. Uma regressão logística pode entrar como complemento interpretável.

### Isso prevê cesárea para uma gestante?

Não é essa a nossa proposta. O SINASC é retrospectivo, então a base registra eventos depois que o parto aconteceu. Nossa proposta é analisar fatores associados em escala populacional.

### Robson não é circular?

Ele é parcialmente redundante com variáveis obstétricas, sim. Por isso separamos Robson como estratificação clínica, não como causa isolada.

### Por que remover profissional assistente se aumentava AUC?

Porque ele descreve o processo assistencial no momento do parto. Para a nossa pergunta sobre determinantes anteparto, ele aumenta desempenho, mas piora a interpretação.

### O que significa estabelecimento ser tão importante?

Sugere heterogeneidade institucional: diferentes unidades podem ter práticas e perfis de atendimento distintos. Isso não prova causalidade do hospital, mas indica que o contexto institucional concentra muito sinal.
