# dashboard

Página única (`index.html`) para comparar o real com o previsto por setor e
ciclo a partir dos **runs** arquivados pelo backtest
(`experiments/compare_forecasters/run.py`, ver
[guia de execuções](../docs/guides/forecasting-backtest-executions.md)).
As métricas seguem a análise de Prophet da Layza (MAE, WMAPE, Bias; branch
`feat/processamento-dados`), mais RMSE e P90 do erro absoluto.

## Como usar

A página é um arquivo HTML comum: não precisa de servidor nem de Python, só
de um navegador (Chrome, Edge ou Firefox) e de internet para baixar as
bibliotecas de gráfico (Plotly e PapaParse, do cdnjs).

### 1. Gerar os runs

Os runs ficam em `experiments/compare_forecasters/outputs/`, que o git ignora.
Por isso, cada pessoa gera os seus depois de clonar o repositório. Na raiz:

```bash
uv sync --extra forecast
uv run python -m experiments.compare_forecasters.run configs/experiments/compare_forecasters/runs_v2_referencias.yaml
uv run python -m experiments.compare_forecasters.run configs/experiments/compare_forecasters/runs_v2_organizacao.yaml
```

No Windows, com o `.venv` já instalado:

```bat
.\.venv\Scripts\python.exe -m experiments.compare_forecasters.run configs/experiments/compare_forecasters/runs_v2_referencias.yaml
```

Cada run leva cerca de 2 minutos e cria uma pasta nova com `predictions.csv`,
`comparison.csv`, `config.yaml` e `manifest.json`. Para um teste novo, copie
um YAML e troque `paths.run_dir`: o runner recusa uma pasta que já existe, em
vez de sobrescrevê-la.

### 2. Abrir a página

- Linux: `xdg-open dashboard/index.html`
- macOS: `open dashboard/index.html`
- Windows: `start dashboard\index.html`

Duplo clique no arquivo, ou arrastá-lo para uma aba do navegador, também
funciona.

### 3. Carregar a pasta de runs

Em **Pasta de runs**, clique em **Escolher arquivos** e selecione a pasta
`experiments/compare_forecasters/outputs` em si, sem entrar nela. O navegador
pergunta se pode "fazer upload" dos arquivos: pode aceitar. A página só lê os
arquivos no próprio navegador, e nada é enviado para a internet.

Toda subpasta com `predictions.csv` vira um run, e os CSVs soltos fora de uma
pasta de run são ignorados. O run completo mais recente vem marcado; marque
outros na tabela de runs para comparar modelos entre runs, até 8 modelos ao
mesmo tempo. Depois de gerar um run novo, escolha a pasta de novo para
recarregar.

### 4. Ler os gráficos

- **Real vs previsto por ciclo**: a linha preta é o real e as coloridas são
  os modelos. Quanto mais perto da preta, melhor.
- **Métrica por ciclo**: mostra em quais ciclos cada modelo erra mais. Em
  Bias, acima de zero o modelo previu demais e abaixo de zero previu de menos.
- **Real × previsto por ponto**: cada ponto é um setor num ciclo, e a
  diagonal é o acerto exato. Clicar num ponto filtra a página por aquele setor.
- **Métrica por grupo**: o erro por região, gerência, CD ou UF.
- **Setores com pior métrica**: clicar numa linha filtra aquele setor.

O botão **Limpar filtros** volta à visão completa.

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
