# reports_dashboard

Dashboard estático com os resultados dos experimentos de forecast de nível
(setor/cidade/ciclo). Compara os candidatos de cada experimento por MAE,
WMAPE, viés e o trade-off MAE × viés.

## Como usar

Abra `index.html` no navegador (duplo-clique ou arraste pro browser). Não
precisa de servidor — os dados estão embutidos no HTML e os gráficos usam
Chart.js via CDN (requer internet para carregar a biblioteca).

No topo há três seletores:

- **Experimento** — qual dos 6 experimentos ver.
- **Estado** — "Todos os estados" (agregado Brasil) ou um UF específico.
- **Cidade** — "Todas" ou uma cidade. A lista de cidades é filtrada pelo
  estado escolhido; com estado "Todos", mostra as 120 cidades de maior volume.

As métricas (MAE, WMAPE, viés, RMSE) são **recalculadas** para o recorte
geográfico escolhido, a partir dos erros linha a linha de `predictions.csv`.
A tabela ordena ao clicar nos cabeçalhos; a melhor linha por MAE fica
destacada.

Observação: o MAE aqui é sobre todos os pontos de cada candidato no recorte,
não o "subconjunto comum" do `comparison.csv` — por isso os valores diferem
um pouco dos relatórios, embora o ranking entre candidatos se mantenha.

## Experimentos incluídos

1. Features + temporais (base completa)
2. Externas por UF (censo/salário/desemprego)
3. Momentum + calendário
4. Prophet (amostra de 60 setores)
5. Croston / SBA / TSB (demanda intermitente)
6. Objetivos Tweedie / Quantile

## Como regenerar os dados

Os dados por estado/cidade são pré-agregados dos `predictions.csv` em
`experiments/compare_forecasters/outputs/2026-10-19_*` pelo script
`scripts/build_dashboard_data.py`, que gera `reports_dashboard/_data.json`.
Depois, o conteúdo desse JSON é colado dentro da tag `<script id="data">` no
`index.html` (substituindo o placeholder `__DATA__`).
