# Backtest hierárquico de level

Este fluxo compara distribuição hierárquica e reconciliação, sem alterar os
executores anteriores de level, a integração diária ou o shape. A saída é
volume de itens por ciclo. Os níveis disponíveis são região (`CD_RE`),
gerência (`CD_RE`, `CD_GV`), setor (`cd_setor`) e cidade
(`cd_setor`, `estado`, `cidade`).

## Executar

Na raiz do repositório, com o ambiente virtual já preparado:

```powershell
.\.venv\Scripts\python.exe -m experiments.hierarchical_level.run configs/experiments/hierarchical_level/smoke.yaml
```

`smoke.yaml` avalia as duas últimas origens elegíveis. A comparação padrão
tem somente três candidatos: média móvel pura de cinco ciclos,
ETS + distribuição LightGBM e ETS + LightGBM reconciliado. O histórico
continua sendo de 20 ciclos fechados. Para avaliar todas as origens:

```powershell
.\.venv\Scripts\python.exe -m experiments.hierarchical_level.run configs/experiments/hierarchical_level/full.yaml
```

A média móvel é fixa em cinco registros: não há seleção de janela nem
modo histórico no código atual. Resultados e parâmetros antigos continuam
arquivados; o código anterior à limpeza está no commit `69b9ece`.
Escolha uma pasta de saída nova em qualquer repetição.

Em um ambiente novo, instale Python 3.13 e as dependências do projeto, por
exemplo `uv sync --extra forecast --group dev`. O SciPy é usado na
reconciliação; nenhum solver de atribuição ou licença Gurobi é necessário.

## Um arquivo de configuração

Para somar setores e prever diretamente por gerência–cidade, veja o
[experimento de gerência e cidade](hierarchical-management-city.md).

Edite o YAML e escolha uma pasta nova em `paths.run_dir`. Uma execução não
sobrescreve uma pasta existente. `paths.base_csv` aponta para a base de
demanda com os mesmos campos do forecasting atual, incluindo organização,
estado e cidade.

| Campo | Significado |
|---|---|
| `levels` | Níveis em ordem, terminando em `city`; é possível começar em `management` ou `sector`, ou pular um nível intermediário |
| `city_grouping` | `sector` por padrão; `management` agrega a cidade na gerência e exige gerência imediatamente antes da cidade |
| `history_cycles` | Quantidade de ciclos globais completos e fechados usados no treino; mínimo de cinco, padrão de 20 |
| `share_window` | Janela recente das participações históricas usadas como referência dos modelos |
| `ets_alphas` | Pesos de suavização candidatos; seleção apenas no treino |
| `learning.features` | Blocos de features habilitados no LightGBM |
| `learning.city_prior_cycles` | Regularização: quanto maior, mais as cidades usam a referência histórica |
| `learning.volume_objective` | Objetivo das previsões diretas internas; padrão `tweedie` |
| `max_origins` | Limita às últimas N origens elegíveis; omitir para avaliar todas |
| `first_target_cycle`, `last_target_cycle` | Limites inclusivos de ciclos avaliados |
| `scenarios_csv` | Opcional: arquivo de cenários completos de abertura |

Cada nível interno possui um LightGBM compartilhado entre seus nós. Não se
treina um LightGBM independente para cada cidade. A configuração dos modelos
de ML é compartilhada entre os níveis internos nesta primeira versão.
Escreva os limites de ciclo entre aspas no YAML, por exemplo
`first_target_cycle: "202604"`.

## Candidatos e baselines

Somente os três candidatos abaixo entram nas previsões e métricas:

- `direct_ma_5`: média dos últimos cinco registros disponíveis de cada nó,
  incluindo zeros. Sem distribuição ou reconciliação; pode não somar entre
  níveis quando o cadastro/histórico é desigual. Séries curtas usam os
  registros disponíveis, sem inventar observações.
- `distribution_ets_lgbm`: ETS no nível inicial; LightGBM estima participações
  nos internos. Cada grupo de irmãos soma 100%, preservando o total superior.
- `reconciliation_ets_lgbm`: ETS no nível inicial e LightGBM de volume nos
  internos; todos os níveis são ajustados conjuntamente.

Os campos antigos `comparison` e `baseline_windows` não são mais aceitos
nos YAMLs de execução. A janela fixa aparece como `baseline_window: 5` em
`features.json`. Médias móveis usadas como features ou como regularização
dos modelos continuam presentes; são entradas diferentes do candidato
de referência.

O ETS é simples, sem componente sazonal anual. Seu alpha e o expoente de
duração (0, 0,5 ou 1) são escolhidos pelos erros de suavização do histórico.
Ele utiliza itens e duração, não a tabela completa de features do ML.

A reconciliação é mínimos quadrados ponderados não negativos. Minimiza as
diferenças entre previsões coerentes e previsões base, divididas pelo
desvio-padrão histórico de cada nó (mínimo de um item). A previsão de região
também pode mudar. Não é MinT: não se estima uma matriz de covariâncias dos
erros. A matriz de somas é esparsa e o procedimento falha explicitamente se
não houver convergência. O objetivo de reconciliação é quadrático, ainda que
a comparação operacional também reporte MAE e WMAPE.

## Features implementadas

`features.json` registra o catálogo e a seleção efetiva. Identificadores
`node` e `parent` são categóricos em todos os modelos internos. Variáveis
numéricas não recebem Min–Max nem padronização.

| Bloco | Variáveis |
|---|---|
| `history` | Lags 1, 2, 3; médias 3, 5, 10; desvio-padrão 5; média 3 / média 10 |
| `calendar` | Mês da abertura global do ciclo, número do ciclo, duração média/mínima/máxima das janelas, dia da semana médio e dia do mês médio das aberturas, dias desde a última abertura histórica disponível |
| `structure` | Quantidade conhecida de filhos, quantidade recente de filhos com compra, mudança na quantidade de filhos conhecidos |
| `hierarchy` | Participação histórica do filho no pai em 5 e 10 registros, média histórica do pai, participação anterior do maior filho |
| `city_recent_shares` | Opcional: `share_1`, `share_2`, `share_3`, participação da cidade no pai configurado nos últimos 1, 2 e 3 registros fechados; ausentes nos demais níveis |
| `intermittency` | Frequência de compra em 5 registros, quantidade de registros fechados desde compra anterior, média de compras positivas, quantidade de observações disponíveis |

As janelas são registros anteriores disponíveis de cada nó. Na cidade,
incluem zeros após a primeira compra conhecida. Um nó recente pode ter menos
de K observações. Sem compras positivas, a média positiva fica ausente;
LightGBM trata ausência numericamente. A participação é razão entre somas,
não média de razões. Denominadores zero ficam ausentes nos preditores; pais
sem volume não fornecem alvos de participação ao treinamento.

Para testar as participações recentes, acrescente `city_recent_shares` à lista
`learning.features`, mantendo os demais blocos. O bloco não é habilitado por
padrão, preservando as configurações existentes. Cada `share_K` é a soma dos
itens da cidade dividida pela soma do setor nos mesmos K registros históricos
disponíveis, com zeros de cidades conhecidas e `NaN` quando o denominador é
zero. O exemplo atual não entra no cálculo, e ciclos ainda abertos são excluídos.

A configuração `configs/experiments/hierarchical_level/recent_city_shares.yaml`
repete os parâmetros de `full.yaml` e acrescenta somente esse bloco, gravando
em uma nova pasta:

```powershell
.\.venv\Scripts\python.exe -m experiments.hierarchical_level.run configs/experiments/hierarchical_level/recent_city_shares.yaml
```

Para repetir uma execução, escolha outro `paths.run_dir`. O acréscimo afeta os
LightGBMs de cidade. Na distribuição, os totais dos pais permanecem iguais;
na reconciliação, mudanças nas cidades também podem ajustar os níveis superiores.

O resultado da comparação controlada de 24/10/2026 está em
[Participação recente das cidades](../reports/hierarchical-city-recent-shares.md).
Não houve ganho material no nível cidade; o bloco permanece opcional.

As quantidades de componentes são inferidas das compras históricas
disponíveis, não de um cadastro comercial oficial. Por isso, não representam
necessariamente todas as cidades ou setores que a organização poderia atender.
Na consulta futura, a quantidade de filhos vem da estrutura histórica
conhecida e dos setores planejados no cenário.

Nos níveis agregados, duração e características de abertura são resumidas
primeiro por setor, com pesos iguais, evitando dar mais peso a um setor
apenas porque possui mais cidades. Uma região não recebe uma abertura
individual fictícia. Esta versão não inclui feriados, top 3 componentes ou
ponderação da composição do cenário pelo volume histórico.

## Histórico, cidades novas e organização

Cada origem usa somente ciclos completos cujo fechamento global é anterior
à primeira abertura global do alvo. Registros de pedidos posteriores ao
instante simulado também são excluídos. Essa restrição é aplicada tanto ao
histórico da origem quanto às features de cada exemplo histórico do ML.

O cadastro de cidades consultadas vem do passado. Uma cidade conhecida que
não compra no alvo continua prevista e no diagnóstico completo, mas não entra
nas métricas principais de cidade. Uma cidade que
aparece pela primeira vez no alvo não entra no treinamento ou cadastro de
previsão: sua previsão de avaliação é zero, seu erro é contado e sua falta
de cobertura é explícita. Uma nova cidade nunca é inventada para melhorar
o cadastro retroativamente.

O cadastro usado nesta primeira versão é o recuperável no painel de treino.
Cidades observadas antes da janela continuam presentes pelos zeros quando
seu setor aparece na janela. Um setor sem qualquer registro nos ciclos
selecionados fica sem previsão própria; seu volume também conta no erro e
na falta de cobertura. Um cadastro comercial independente permite ampliar
esse tratamento em uma evolução posterior.

Para cidades com poucas compras, combina-se o score do ML com a referência
histórica, com peso `compras_positivas / (compras_positivas + prior)`. Depois,
as participações são normalizadas. Se todos os scores de irmãos forem zero,
usa-se distribuição uniforme e não um denominador fictício.

O vínculo organizacional da consulta é o último conhecido nos ciclos
fechados. Na avaliação, setores conhecidos são agrupados por esse vínculo;
metadados reais de um setor totalmente novo servem somente para avaliar seu
volume, nunca para gerar features. Mudanças futuras de cadastro exigem um
cadastro planejado explícito antes de usar o modelo operacionalmente.

## Cenários e métricas

Por padrão, a análise principal de cidade considera apenas combinações
`setor–estado–cidade–ciclo` com `actual > 0`. Neste relatório, "ativa" significa
que houve demanda naquele ciclo, não um status cadastral conhecido antecipadamente.
`summarize()` em `hierarchy_evaluation.py` aplica esse filtro apenas depois de
gerar as previsões. Treinamento, features, cadastro consultado e distribuição
mantêm os zeros históricos. Não se redistribui o volume usando compradores
conhecidos depois do ciclo. Região, gerência e setor continuam avaliados com
os totais originais e todas as suas previsões.

As métricas principais são condicionais à compra: não penalizam demanda
prevista em cidades sem compra. Os arquivos `*_all_points.csv` conservam essa
avaliação completa como diagnóstico. Cidades novas com compra continuam
contando como erro e falta de cobertura. Grupos sem cidades ativas aparecem
com `n_points=0` e métricas ausentes, evitando um falso erro zero.

Sem `scenarios_csv`, cada alvo recebe o cenário `historical`, com as janelas
efetivamente observadas. Essas janelas são tratadas como um plano conhecido
antes da previsão. Elas vêm dos registros de demanda e não de um arquivo de
planejamento independente: confirme essa hipótese antes do uso operacional.

Um CSV opcional deve conter uma linha por cenário, ciclo e setor:

```csv
scenario_id,CICLOS,cd_setor,window_start,cycle_days
scenario_a,202612,3257,2026-07-09,21
scenario_a,202612,3258,2026-07-10,21
```

Forneça todos os setores planejados para cada cenário. O exemplo acima é
apenas de formato. O mês global vem do calendário do ciclo; as aberturas e
durações dos setores vêm do CSV. Cenários com a mesma configuração
histórica podem receber métricas. Cenários alternativos são exportados com
`is_observed=false`, `actual` ausente e sem métricas: o histórico não contém
a demanda contrafactual correspondente. Esta implementação não demonstra
efeitos causais de mudar uma abertura.

| Métrica | Definição |
|---|---|
| `mae` | Média do erro absoluto de cada nó e depois média entre nós, com peso igual |
| `mae_pooled` | Erro absoluto médio de todos os pontos |
| `wmape_pct` | 100 × soma dos erros absolutos / soma da demanda real |
| `bias` | Erro médio previsto menos real, em itens |
| `bias_pct` | 100 × soma(previsto − real) / soma(real) |
| `n_missing` | Pontos sem previsão no cadastro conhecido |
| `n_total_points` | Pontos observados no grupo antes do filtro de atividade |
| `n_excluded_points` | Pontos retirados das métricas principais por demanda zero na cidade |
| `scoring_scope` | `city_actual_positive` na análise principal de cidade; `all_points` nos demais casos |
| `coverage_points_pct` | Percentual de pontos com previsão |
| `coverage_volume_pct` | Percentual da demanda real em pontos com previsão |

Demandas de nós novos contam como erro, em vez de serem removidas da
comparação. WMAPE e Bias percentuais ficam ausentes quando o total real é
zero. Compare candidatos no mesmo nível. Não compare diretamente o MAE da
região com o MAE da cidade.

## Arquivos gerados

| Arquivo | Conteúdo |
|---|---|
| `comparison.csv` | Métricas por candidato e nível, em todas as origens observadas |
| `metrics_by_cycle.csv` | As mesmas métricas por ciclo e cenário observado |
| `comparison_all_points.csv` | Diagnóstico com todas as cidades, inclusive demanda zero |
| `metrics_by_cycle_all_points.csv` | Diagnóstico completo por ciclo e cenário |
| `predictions.csv` | Real, previsto, erro, nível, identidade, cenário e cobertura |
| `folds.csv` | Data da previsão e lista exata dos ciclos fechados de treinamento |
| `config.yaml` | Cópia da configuração solicitada |
| `features.json` | Catálogo, seleção e famílias de modelos |
| `fitted_models.json` | Alpha/expoente ETS, importância das features e convergência da reconciliação |
| `manifest.json` | Parâmetros efetivos, hashes, versão do código e status |

Nas novas execuções, `predictions.csv` mantém todos os pontos e acrescenta
`included_in_primary_metrics`, indicando a inclusão na análise principal.
O manifesto registra a definição de atividade e os dois escopos de avaliação.

Os resultados históricos com várias janelas são exploratórios. Não há seleção automática
de vencedor nem ajuste de hiperparâmetros no teste. Para uma avaliação final,
congele parâmetros após o desenvolvimento e reserve origens posteriores em
outra execução com limites de ciclo explícitos.

## Execução de referência de 24/10/2026

Esta seção descreve a versão anterior à limpeza, preservada no commit
`69b9ece`; não é o conjunto de candidatos executado atualmente.
Foi executado o `full.yaml` daquela versão, com os 20 ciclos fechados anteriores disponíveis,
quatro janelas de média móvel e os cinco blocos originais de features
(`history`, `calendar`, `structure`, `hierarchy`, `intermittency`).
Foram nove origens, de 202604 a 202612, com 15 candidatos e 1.592.745 linhas de
previsão. Os resultados estão em
`experiments/hierarchical_level/outputs/2026-10-24_full_v02/`.

Os resultados originais abaixo foram calculados antes da mudança para
cidades ativas e incluem demanda zero. Não compare diretamente seus MAEs
com os do novo escopo. As reapurações sem retreino ficam em
`active_city_analysis/` dentro de cada pasta de execução; os arquivos originais
permanecem preservados.

| Nível | Menor WMAPE observado entre os candidatos | MAE | WMAPE | Bias |
|---|---|---:|---:|---:|
| Região | Reconciliação ETS + LightGBM | 36.805,4 | 10,73% | −0,55% |
| Gerência | Reconciliação ETS + LightGBM | 11.094,4 | 14,95% | −0,55% |
| Setor | Distribuição ETS + LightGBM | 1.850,8 | 41,09% | −6,52% |
| Cidade | Distribuição com média móvel de 3 ciclos | 168,2 | 71,46% | −9,34% |

Na cidade, a reconciliação ETS + LightGBM teve MAE 169,6, WMAPE 72,24% e
Bias −0,55%. Portanto, não houve uma abordagem vencedora em todos os critérios.
A escolha da melhor janela nessa tabela é exploratória sobre o próprio teste.
Os valores estão na escala dos itens mascarados da base, e não são
diretamente comparáveis aos testes anteriores que usavam outro corte de
disponibilidade ou removiam cidades sem previsão.

O arquivo adicional `checks.json` registra a conferência independente dos
fechamentos, das somas entre níveis e das métricas exportadas. A tentativa
`full_v01` foi interrompida para otimização e não representa um teste concluído.

## Reapuração por cidades ativas

As previsões originais das execuções `2026-10-24_full_v02` e
`2026-10-24_recent_city_shares_v01` foram reapuradas em 24/10/2026 sem retreino.
Cada pasta contém `active_city_analysis/`, com `comparison.csv`,
`metrics_by_cycle.csv`, `comparison_all_points.csv` e um manifesto que aponta
para as previsões e parâmetros originais. Foram mantidos 43.951 pontos de
cidades com compra e excluídos 56.318 pontos de demanda zero por candidato.

Resultados da execução com participações recentes, no novo escopo:

| Modelo | MAE | WMAPE | Bias |
|---|---:|---:|---:|
| ETS + distribuição LightGBM | 220,31 | 57,0424% | −21,7222% |
| ETS + LightGBM reconciliado | 221,86 | 57,1988% | −15,6743% |
| Média móvel 20 + reconciliação | 221,20 | 57,3036% | −9,0433% |

A distribuição LightGBM apresenta os menores MAE e WMAPE entre os candidatos
nesse recorte, enquanto a média móvel reconciliada de 20 ciclos tem menor
subestimação que esses dois híbridos. A escolha da melhor janela continua
exploratória. Não compare MAE 220,31 diretamente ao MAE 169,96 da avaliação
com zeros: a população de pontos mudou. Redução de WMAPE por mudar o escopo
de avaliação não significa que o modelo tenha aprendido ou melhorado.

## Arquivos e funções

- `hierarchy_config.py`: `HierarchySettings` valida a configuração.
- `hierarchy_data.py`: `aggregate_nodes` constrói os níveis; `known_roster`
  e `scenario_panel` constroem a consulta conhecida.
- `hierarchy_features.py`: `build_features` e `training_and_query` aplicam os
  cortes de disponibilidade dos rótulos.
- `hierarchy_models.py`: `ets_total`, `historical_shares` e `learned_values`
  geram previsões ou scores.
- `hierarchy_reconciliation.py`: `distribute` e `reconcile` combinam os níveis.
- `hierarchy.py`: `forecast_hierarchy` calcula os candidatos de um cenário.
- `hierarchy_evaluation.py`: `evaluate_hierarchy` executa as origens fechadas.
- `experiments/hierarchical_level/run.py`: `run` grava a execução.
