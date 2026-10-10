# Shape diário: configuração e backtest

Para explorar os resultados em gráficos e filtros, veja o
[front independente de shape](../../shape_dashboard/README.md).

Este fluxo aprende **a distribuição por setor–ciclo–dia**, somando todos os CDs.
Para este teste de shape, use o executor e o YAML abaixo.

## 1. Configure um único arquivo

Abra `configs/experiments/compare_shapes/sector_day.yaml`:

```yaml
split:
  min_train_cycles: 20
  horizon: 1
  window: expanding

candidates:
  - name: uniform
    model: uniform
  - name: sector_10_bins
    model: sector
    n_bins: 10

paths:
  base_csv: data/base_tratada_v2.csv
  run_dir: experiments/compare_shapes/outputs/shape_v01
```

- `min_train_cycles`: quantidade mínima de ciclos globais anteriores à origem,
  não uma exigência de 20 observações para cada setor.
- `window: expanding`: começa com 20 ciclos e aumenta o histórico em cada origem.
  `sliding` mantém os últimos 20 ciclos.
- `horizon: 1`: avalia o próximo ciclo de cada origem. Valores maiores avaliam os
  próximos N ciclos, sem usar as demandas deles no treinamento. Um mesmo ciclo
  pode aparecer em origens diferentes; `origin_cycle` distingue as previsões.
- `base_csv`: caminho da base, relativo à raiz do repositório.
- `run_dir`: pasta exclusiva da execução. Escolha outro nome a cada teste,
  por exemplo `shape_v02_bins5`. O executor recusa sobrescrever uma pasta existente.
- `name`: nome único para identificar o candidato nos CSVs.

## 2. Modelos disponíveis e entradas

`src/forecasting/shape_daily.py → predict_shape` implementa:

| Modelo | Como distribui a demanda | Parâmetros/entradas |
| --- | --- | --- |
| `uniform` | Mesma participação em todos os dias: `1 / n_days` | Abertura e duração da janela futura |
| `sector` | Aprende a curva média histórica do setor em faixas de posição relativa | Setor, participações históricas, abertura, duração e `n_bins` |

Para `sector`, `src/forecasting/shape_daily.py → _learn_bins` usa
`bin = floor(offset * n_bins / n_days)`, soma a participação de cada faixa e
calcula sua média entre ciclos de demanda positiva, com peso igual por ciclo.
`_sector_weights` divide a massa da faixa pelos dias da janela previstos nessa
faixa e normaliza a curva para somar 1. Janelas de durações diferentes podem ser
comparadas. Se a projeção não tiver massa utilizável, usa distribuição uniforme;
`uniform_fallback` sinaliza esse caso. `n_bins` não altera o modelo `uniform`.

Há também um modelo `lightgbm` específico de shape com features de região,
gerência, dia da semana e mês. Veja a
[referência de features](shape-feature-reference.md) e o YAML
`configs/experiments/compare_shapes/calendar_features.yaml`.
ETS, ARIMA e Prophet não estão implementados neste executor de shape.
Alterar `n_bins` compara resoluções do modelo `sector`; no LightGBM, use
`learning.features` para escolher as novas variáveis.

## 3. Execute no terminal

Abra o terminal na raiz do repositório. Com o ambiente virtual e as dependências
do projeto já instalados, execute no Windows:

```bat
.venv\Scripts\python.exe -m experiments.compare_shapes.run configs/experiments/compare_shapes/sector_day.yaml
```

O comando imprime a pasta absoluta e as métricas ao terminar. Não executa solver
nem ajusta modelos de level.

## 4. Entenda os dados avaliados

`src/forecasting/shape_daily.py → build_shape_days` usa o carregador existente
`src/forecasting/dataset.py → load_demand_base`, incluindo sua regra de remoção
das cópias por abertura. Portanto, continuam necessárias as colunas exigidas por
esse carregador. A coluna `cd_cd` não é necessária para o novo shape.

`build_shape_days` considera apenas ciclos completos conforme
`dataset.py → cycle_calendar`. Para cada setor–ciclo observado, a janela vai da
menor abertura ao maior fechamento. Inclui todos os dias corridos dessa janela,
inclusive dias sem pedidos (demanda zero); soma a demanda de todos os CDs.
As datas devem estar sem horários, e os itens precisam ser finitos e não negativos.
Pedidos fora de sua janela declarada geram erro em vez de serem descartados.

`shape_evaluation.py → evaluate_shapes` ordena os ciclos pela abertura e avança
as origens. Não impõe encerramento do histórico antes da primeira abertura do
alvo neste backtest. Usa as janelas históricas reais dos ciclos avaliados; não
gera cenários alternativos de abertura. Setores–ciclos ausentes da base não são
inventados. Um setor sem curva histórica positiva recebe o fallback uniforme.

## 5. Leia os quatro arquivos da pasta

| Arquivo | Conteúdo |
| --- | --- |
| `comparison.csv` | Métricas e contagens por candidato |
| `predictions.csv` | Uma linha por candidato–origem–setor–ciclo–dia, sem CD |
| `config_snapshot.yaml` | Cópia exata da configuração utilizada |
| `run_manifest.json` | Parâmetros efetivos, modelos, entradas usadas, hash da base, revisão Git, alterações locais, versões e status da execução |

`shape_evaluation.py → _score_origin` calcula:

```text
actual_share = actual_do_dia / total_real_do_setor_ciclo
items_pred = share_pred * total_real_do_setor_ciclo
error = items_pred - actual
```

**O total real é usado somente para avaliar o shape em itens.** Não é um level
previsto e não entra na aprendizagem da curva do ciclo alvo. Por exemplo, para
total real 1.000 e participações previstas `[0,20; 0,30; 0,50]`, a avaliação usa
`[200; 300; 500]` itens. Para uso futuro, multiplicaria essas mesmas participações
pelo level previsto, fornecido pelo seu fluxo existente.

`shape_evaluation.py → compare_shapes` ordena por `share_mae_pp`, menor é melhor.
`_candidate_metrics` reporta:

- `share_mae_pp`: média do erro absoluto diário de participação, primeiro em
  cada setor–ciclo–origem e depois entre essas curvas, em pontos percentuais.
- `total_variation`: média de `0,5 * soma(abs(share_pred - actual_share))` por
  curva. Vai de 0 a 1 e mede quanto da distribuição diária está deslocada.
- `mae`, `rmse`: erros em itens, considerando todas as linhas diárias avaliadas.
- `wmape`: soma dos erros absolutos em itens / soma dos itens reais; é razão,
  por exemplo `0,25` significa 25%.
- `bias`: média de `items_pred - actual`; `bias_pct`: 100 vezes soma desse erro
  / soma dos itens reais. Como a curva conserva o total real, o bias agregado
  tende a zero por construção, mesmo se a distribuição diária estiver ruim.
- `n_days`, `n_sector_cycles`, `n_origins`: tamanho da avaliação por candidato.
  `n_sector_cycles` conta combinações de origem–setor–ciclo.
- `n_uniform_fallback_cycles`: curvas de `sector` que usaram fallback.
- `n_zero_total_cycles`: curvas com total zero. Ficam no CSV e nas métricas em
  itens, mas são excluídas das métricas de participação (share indefinido).
  Com denominador zero, métricas percentuais ficam vazias/NaN.

Para comparar testes, mantenha base e `split` iguais. Compare `sector_5_bins`
e `sector_10_bins` na mesma execução adicionando os dois candidatos ao YAML.

## 6. Usar a curva em uma previsão futura

`src/forecasting/shape_daily.py → predict_shape` também aceita janelas futuras:

```python
import pandas as pd
from src.forecasting.dataset import load_demand_base
from src.forecasting.shape_daily import build_shape_days, predict_shape

history = build_shape_days(load_demand_base("data/base_tratada_v2.csv"))
# Em produção, filtre o histórico para informações disponíveis antes da previsão.
windows = pd.DataFrame({
    "cd_setor": ["SEU_SETOR"], "CICLOS": ["CICLO_FUTURO"],
    "window_start": [pd.Timestamp("2027-01-05")], "n_days": [10],
})
curve = predict_shape(history, windows, model="sector", n_bins=10)
# Após juntar seu level previsto por cd_setor/CICLOS:
# items_pred = share_pred * level_previsto
```

Esta API recebe as datas de abertura e durações escolhidas; não as descobre.
Ela retorna `share_pred` por dia sem CD. Não altera o executor futuro de level
nem substitui automaticamente o antigo fluxo `daily:`.
