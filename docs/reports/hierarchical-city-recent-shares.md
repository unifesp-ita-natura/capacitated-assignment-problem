# Participação recente das cidades: comparação controlada

Data: 24/10/2026.

## Alteração avaliada

Adição opcional de `city_recent_shares`: `share_1`, `share_2` e `share_3`.
`_hierarchy_features()` em `src/forecasting/hierarchy_features.py` calcula a
razão entre a soma dos itens da cidade e a soma do setor nos mesmos registros.
`_closed_history_features()` aplica o corte de ciclos já encerrados antes da
abertura de cada exemplo. As novas colunas ficam ausentes nos outros níveis.
O cálculo mantém os zeros de cidades conhecidas e produz ausência quando a
demanda do pai é zero. O bloco precisa constar em `learning.features` para
entrar nos modelos; as seleções anteriores e os executores de shape não mudam.

Referência: `experiments/hierarchical_level/outputs/2026-10-24_full_v02/`.
Experimento: `experiments/hierarchical_level/outputs/2026-10-24_recent_city_shares_v01/`.
Configuração: `configs/experiments/hierarchical_level/recent_city_shares.yaml`.

Foram mantidos os nove alvos de 202604 a 202612, os 20 ciclos globais fechados
de treino, a mesma base, os hiperparâmetros e os 15 candidatos. Cada candidato
foi avaliado nos mesmos 100.269 pontos de cidade. As 12 variantes de médias
móveis permaneceram iguais em todos os níveis, dentro da tolerância de 1e-9.
Hashes de dados, parâmetros efetivos, folds e cobertura foram conferidos.

## Resultado no nível cidade, incluindo demanda zero

| Modelo | MAE antes | MAE depois | WMAPE antes | WMAPE depois | Bias antes | Bias depois |
|---|---:|---:|---:|---:|---:|---:|
| ETS + distribuição LightGBM | 169,9465 | 169,9640 | 72,2353% | 72,2421% | −6,5225% | −6,5225% |
| ETS + LightGBM reconciliado | 169,5647 | 169,5128 | 72,2365% | 72,2139% | −0,5517% | −0,6592% |
| ETS + LightGBM direto, diagnóstico | 169,6269 | 169,4985 | 72,2642% | 72,2088% | −1,9622% | −2,2237% |
| Média móvel 3 + distribuição histórica | 168,1920 | 168,1920 | 71,4636% | 71,4636% | −9,3354% | −9,3354% |

O MAE é a média dos erros médios por nó, conforme `_metrics()` em
`src/forecasting/hierarchy_evaluation.py`. Não é `mae_pooled`.
Bias negativo indica subestimação. Reduzir seu valor absoluto é desejável.

A distribuição piorou 0,0068 ponto percentual de WMAPE. A reconciliação
melhorou 0,0226 ponto percentual, redução relativa de apenas 0,0312%, mas seu
Bias ficou mais negativo. Ela melhorou WMAPE em quatro dos nove ciclos; a
distribuição melhorou em três. Não há melhora consistente ou material na cidade
neste teste. A média móvel distribuída de três ciclos continua melhor em MAE
e WMAPE completos, embora tenha maior subestimação agregada.

## Efeito nos níveis superiores

As previsões de distribuição e as previsões diretas de região, gerência e
setor permaneceram iguais. `reconcile()` em
`src/forecasting/hierarchy_reconciliation.py` ajusta todos os níveis em conjunto,
portanto as mudanças nas previsões de cidade alteraram os agregados reconciliados:

| Nível reconciliado | WMAPE antes | WMAPE depois | Redução relativa |
|---|---:|---:|---:|
| Região | 10,7267% | 10,6392% | 0,8154% |
| Gerência | 14,9544% | 14,9187% | 0,2389% |
| Setor | 41,3396% | 41,3058% | 0,0818% |

O Bias agregado passou de −0,5517% para −0,6592% em todos os níveis reconciliados.

## Diagnóstico somente onde a demanda real foi positiva

Este filtro foi aplicado depois das previsões, exclusivamente para análise.
Não se removeu nenhuma cidade do cadastro, da distribuição ou do treinamento.

| Modelo | WMAPE antes | WMAPE depois |
|---|---:|---:|
| ETS + distribuição LightGBM | 57,0480% | 57,0424% |
| ETS + LightGBM reconciliado | 57,2072% | 57,1988% |
| Média móvel 3 + distribuição histórica | 57,6494% | 57,6494% |

Nesse recorte os modelos de ML já eram melhores que essa média móvel antes
das novas features. As melhorias adicionais são muito pequenas. Esse recorte
não penaliza previsões positivas nas cidades sem compra.

## Evidências e reprodução

A pasta do experimento contém os arquivos normais da execução e:

- `feature_comparison.csv`: comparação por modelo e nível; deltas são depois − antes.
- `feature_comparison_by_cycle.csv`: comparação por ciclo.
- `feature_comparison_positive_city.csv`: diagnóstico de demanda positiva.
- `comparison_checks.json`: verificações da equivalência das duas execuções.

Para repetir, escolha uma pasta nova em `paths.run_dir` e execute:

```powershell
.\.venv\Scripts\python.exe -m experiments.hierarchical_level.run configs/experiments/hierarchical_level/recent_city_shares.yaml
```

Validação: 18 testes da hierarquia passaram; os seis casos focados em features
e disponibilidade temporal passaram com 92% de cobertura do módulo de features.
Ruff sem erros e complexidade Radon A nas funções verificadas.

Conclusão: manter o bloco disponível para experimentos, sem promovê-lo ao
padrão. Os nove ciclos reutilizados são exploratórios, e não foi demonstrada
significância estatística nem generalização para origens posteriores.
