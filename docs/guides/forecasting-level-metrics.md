# Métricas do backtest de level

O executor `experiments/compare_forecasters/run.py::run` salva as métricas
na tabela indicada por `paths.output_csv` do YAML. Não é necessário ativar
as novas métricas na configuração. O arquivo `paths.errors_csv` continua
contendo os valores reais, previstos e erros absolutos por setor, ciclo e rodada.

`src/forecasting/comparison.py::compare` mantém o ranking de level por
`mae_common`. Todas as métricas com sufixo `_common` usam somente os pontos
(setor, ciclo, rodada) previstos por todos os candidatos. Os totais de cobertura
`n_scored_own`, `n_scored_common` e `n_missing` ajudam a interpretar essa seleção.

| Coluna | Cálculo e interpretação |
|---|---|
| `mae_common` | Média dos erros absolutos de cada setor, seguida da média entre setores. Unidade: itens. |
| `rmse_common` | Raiz da média dos erros ao quadrado sobre todos os pontos comuns. Unidade: itens. |
| `mase_common` | Em cada rodada, MAE dos pontos comuns dividido pelo erro ingênuo calculado no histórico de treinamento; depois, média entre rodadas com pontos comuns. |
| `p90_abs_error_common` | Percentil 90 dos erros absolutos dos pontos comuns. Unidade: itens. |
| `worst_origin_mae_common` | Maior erro absoluto médio por rodada, com peso igual por ponto dentro da rodada. Unidade: itens. |
| `bias_common` | Média de previsto menos real sobre todos os pontos comuns. Positivo indica superestimação; negativo, subestimação. Unidade: itens. |
| `bias_pct_common` | 100 × soma(previsto − real) / soma(real). Já está em percentual: −3,33 significa −3,33%. |
| `wmape_common` | Soma dos erros absolutos / soma(real). Proporção: 0,1667 significa 16,67%. |

Bias e WMAPE são calculados por `src/forecasting/metrics.py::bias`,
`bias_pct` e `wmape`. WMAPE e Bias percentual agregados refletem o volume
total, enquanto `src/forecasting/evaluation.py::equal_weight_mae` dá peso
igual a cada setor. Erros de sinais opostos podem se cancelar no Bias.

`src/forecasting/comparison.py::_common_mase` preserva o denominador de
treinamento de cada rodada. Um denominador zero produz MASE infinito,
conforme `src/forecasting/metrics.py::mase`; um denominador indisponível
produz NaN. Rodadas sem pontos comuns não entram na média do MASE.

Sem pontos comuns, as métricas são NaN. Bias percentual e WMAPE também são
NaN quando a soma da demanda real é zero. Esses casos não têm percentual definido.

A avaliação diária, quando habilitada pelo bloco `daily`, conserva seu ranking
por `mae_cd_day_common`. As novas colunas descrevem os totais de level.
