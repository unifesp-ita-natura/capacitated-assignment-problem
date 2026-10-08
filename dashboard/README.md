# dashboard

Página única (`index.html`) para comparar o real com o previsto por setor e
ciclo a partir dos **runs** arquivados pelo backtest
(`experiments/compare_forecasters/run.py`, ver
[guia de execuções](../docs/guides/forecasting-backtest-executions.md)).
As métricas seguem a análise de Prophet da Layza (MAE, WMAPE, Bias; branch
`feat/processamento-dados`), mais RMSE e P90 do erro absoluto.

## Como usar

1. Gere um ou mais runs: cada YAML com `paths.run_dir` cria uma pasta nova em
   `experiments/compare_forecasters/outputs/`. Exemplos:
   `configs/experiments/compare_forecasters/runs_v2_referencias.yaml` e
   `runs_v2_organizacao.yaml`.
2. Abra `dashboard/index.html` no navegador (Chrome, Edge ou Firefox; precisa
   de internet para carregar Plotly e PapaParse do cdnjs).
3. Em **Pasta de runs**, escolha `experiments/compare_forecasters/outputs`.
   Toda subpasta com `predictions.csv` vira um run; CSVs soltos fora de uma
   pasta de run são ignorados. O run completo mais recente vem marcado; marque
   outros para comparar modelos entre runs (até 8 modelos ao mesmo tempo).

Nada sai do navegador: os arquivos são lidos localmente.

## O que mostra

- **Runs**: status, início, split, hash da base e commit, lidos do
  `manifest.json`. Status diferente de `complete` aparece com ⚠.
- **Métrica**: o seletor no topo escolhe MAE, WMAPE, Bias %, Bias, RMSE ou P90.
  Ele comanda o gráfico por ciclo, o gráfico por grupo, a ordenação das
  tabelas e o ranking de setores (Bias é ordenado pelo valor absoluto).
- **Filtros**: modelo, setor, região (`CD_RE`), gerência (`CD_GV`), CD, UF,
  intervalo de ciclos, horizonte (quando o run tem mais de um) e "só pontos
  previstos por todos os modelos" (ligado por padrão).
- **Gráficos**: real vs previsto por ciclo; a métrica por ciclo; dispersão
  real × previsto (clique num ponto filtra o setor); a métrica por região,
  gerência, CD ou UF.
- **Tabelas**: métricas por modelo no recorte; setores com pior valor da
  métrica (clique filtra o setor); placar oficial de cada run
  (`comparison.csv`, sem os filtros).

Passar o mouse sobre um modelo mostra a família e as features registradas no
`manifest.json`.

## Região, gerência, CD e UF

O runner grava essas colunas em `predictions.csv` a partir de
`src/forecasting/dataset.py::sector_cycle_attributes`. Região e gerência têm
um só valor por setor-ciclo. CD e UF não: cerca de 28% dos setor-ciclos da v2
passam por mais de um CD. Nesses casos o setor-ciclo é atribuído ao CD (e à
UF) com mais itens. Runs gravados antes dessa mudança não têm as colunas, e os
filtros correspondentes aparecem desativados.

## Diferença em relação ao `comparison.csv`

O MAE do painel é a média simples por ponto (setor×ciclo), como na análise da
Layza. O `mae_common` do `comparison.csv` dá peso igual a cada setor
(`src/forecasting/evaluation.py::equal_weight_mae`), então os dois diferem um
pouco. WMAPE, Bias %, RMSE e P90 usam as mesmas definições de
`src/forecasting/metrics.py`. Bias = previsto − real, positivo = superestima (a
Layza usava o sinal oposto).
