# Gerência-cidade: histórico de 10 contra 20 ciclos

Data: 27/10/2026. Critério principal: MAE, acompanhado de WMAPE e Bias.

## Desenho e população comum

Referência reutilizada: `2026-10-27_management_city_v01`, com 20 ciclos.
Nova execução: `2026-10-27_management_city_history10_v01`, com 10 ciclos.
Ambas ficam em `experiments/hierarchical_level/outputs/`.

Foram mantidos os nove alvos 202604 a 202612, as datas de previsão, o hash
da base, versões dos pacotes, seis blocos de features, sementes e demais
hiperparâmetros. O treino curto é exatamente o sufixo de dez ciclos fechados
do treino de vinte em cada origem. Os parâmetros internos do ETS são escolhidos
novamente sobre o respectivo treino. Nenhum algoritmo foi alterado.

`_origin()` em `src/forecasting/hierarchy_evaluation.py` seleciona ciclos
globais encerrados antes da primeira abertura do alvo, e não apenas compras
positivas por cidade. As métricas principais usam os mesmos 38.945 pontos
ativos, com a mesma demanda real. Os 33.930 pontos de demanda zero ficam
excluídos. Os mesmos 726 pontos ativos sem previsão contam como erro em ambos
os históricos; não houve filtragem por interseção dos pontos previstos.

## Modelos e médias móveis distribuídas

| Alternativa | Histórico disponível | Janela da média | MAE | WMAPE | Bias |
|---|---:|---:|---:|---:|---:|
| Média móvel + distribuição histórica | 10 | 10 | 236,91 | 50,7507% | −12,8764% |
| Média móvel + distribuição histórica | 20 | 10 | 233,22 | 49,9738% | −12,9157% |
| Média móvel + distribuição histórica | 20 | 20 | 232,93 | 50,1251% | −5,4041% |
| Compartilhamento ETS + LightGBM | 10 | — | 236,33 | 50,9036% | −16,6307% |
| Compartilhamento ETS + LightGBM | 20 | — | 230,74 | 49,7910% | −17,8930% |
| Reconciliação ETS + LightGBM | 10 | — | 242,36 | 52,0108% | −20,2286% |
| Reconciliação ETS + LightGBM | 20 | — | 235,01 | 50,3531% | −16,3590% |

Reduzir de 20 para 10 ciclos piorou o MAE do compartilhamento em 2,42% e
da reconciliação em 3,12%. Ambos também pioraram em WMAPE. O Bias do
compartilhamento ficou menos negativo com dez; o da reconciliação piorou.

O compartilhamento com vinte continua com o menor MAE e WMAPE nesta
comparação. Sua redução de MAE frente à MA20 distribuída é apenas 0,94%,
enquanto a média móvel apresenta Bias muito mais próximo de zero.
Nenhum dos modelos com dez superou a MA20 em MAE ou WMAPE.

## Médias móveis diretas e interpretação das janelas

| Alternativa | Histórico disponível | MAE | WMAPE | Bias |
|---|---:|---:|---:|---:|
| Média móvel direta de 10 ciclos | 10 | 237,03 | 50,7848% | −12,5387% |
| Média móvel direta de 20 ciclos | 20 | 233,56 | 50,2794% | −4,5869% |

`moving_average()` em `src/forecasting/hierarchy_models.py` usa no máximo K
observações disponíveis por nó, mantendo zeros válidos. A variante direta
não impõe coerência entre níveis. A distribuída usa a média no nível inicial
e reparte por participações históricas suavizadas. `historical_shares()` no
mesmo arquivo mistura as participações recentes com a razão de todo o treino.
Assim, MA10 distribuída com histórico de vinte não é igual à MA10 distribuída
com histórico de dez. O MAE 233,22 anteriormente apresentado corresponde à
primeira e não a uma cadeia inteiramente limitada a dez ciclos.

## Níveis superiores

| Nível | Abordagem | MAE com 20 | MAE com 10 | WMAPE com 20 | WMAPE com 10 |
|---|---|---:|---:|---:|---:|
| Região | Compartilhamento, total ETS | 38.659,38 | 37.245,18 | 11,2670% | 10,8548% |
| Região | Reconciliação | 38.930,87 | 42.827,81 | 11,3461% | 12,4818% |
| Gerência | Compartilhamento | 11.859,48 | 12.102,19 | 15,9856% | 16,3128% |
| Gerência | Reconciliação | 11.826,98 | 12.788,80 | 15,9418% | 17,2383% |

Dez ciclos ajudaram o total regional do ETS isolado, mas não gerência-cidade.
Vinte não é uma regra universal para todos os níveis. Não foi testado um
histórico diferente por nível na mesma execução.

## Evidências e reprodução

Pasta da análise:
`experiments/hierarchical_level/outputs/2026-10-27_management_city_history_comparison_v01/`.

- `comparison.csv`: todos os candidatos, níveis e históricos;
- `city_summary.csv`: principais alternativas de cidade;
- `paired_models.csv`: modelos lado a lado, com diferenças de 10 menos 20;
- `metrics_by_cycle.csv`: resultados por ciclo e histórico;
- `checks.json`: conferência de dados, pacotes, parâmetros, features, folds,
  populações, coberturas e recálculo independente de MAE/WMAPE/Bias;
- `manifest.json`: referências às execuções e ao critério de avaliação.

As duas execuções preservam previsões, configurações, features, parâmetros
efetivos, modelos ajustados e diagnósticos com e sem zeros. Para reproduzir
o teste curto, use `management_city_history10.yaml` conforme o
[guia de gerência-cidade](../guides/hierarchical-management-city.md), alterando
a pasta de saída para não sobrescrever resultados.

Como o algoritmo não mudou, a referência de vinte e a suíte de testes não
foram repetidas. A nova execução validou o YAML e concluiu os nove folds;
a auditoria verificou os resultados. `git diff --check` passou.

Conclusão deste teste: manter vinte ciclos como referência para gerência-cidade
e compartilhamento como menor MAE, conservando a MA20 como referência de menor
Bias. A vantagem é pequena; os nove ciclos já foram usados no desenvolvimento
e não foi demonstrada significância estatística ou generalização para novos alvos.
