# Level: passo a passo para previsões futuras na base real

**Revisão:** 2026-10-05.  
**Escopo:** total de itens por setor-ciclo e janela de abertura. Sem shape, demanda diária,
dados sintéticos ou solver.

Este documento fornece instruções e exemplos. Os scripts, CSVs e configurações dos
exemplos não foram criados nesta tarefa. Nenhuma previsão real foi executada.
O roteiro de avaliação está em [Backtesting e features](forecasting-level-backtesting.md).
Para entender, com exemplos interativos, como a data de abertura e a duração da janela entram
no LightGBM e nos modelos de série, abrir [forecasting-data-de-abertura.html](forecasting-data-de-abertura.html)
no navegador (as fontes vêm do Google Fonts; sem internet, usa as fontes padrão).

## 1. Entender os inputs e a saída

Em src/forecasting/dataset.py::build_item_panel, level é o total de
total_itens_mascarado do setor no ciclo, chamado items.

src/forecasting/model.py::forecast responde a consultas com a chave:

**cd_setor + CICLOS + window_start + cycle_days**

A saída items_pred é o total esperado de itens naquela janela. Datas e durações
são inputs de level, mesmo sem distribuir o total entre os dias.

Usar dataset.py::build_item_panel e model.py::forecast. Não usar src/main.py::main ou
model.py::forecast_future_cycles neste roteiro: esses caminhos combinam level/shape.

## 2. Preparar o ambiente

Executar na raiz:
C:\Users\LeonardoFerreira\Documents\GitHub\capacitated-assignment-problem.

pyproject.toml exige Python >=3.13 e declara LightGBM/statsmodels no extra forecast.

~~~powershell
uv sync --extra forecast
~~~

O comando prepara o ambiente; não executa solver.
Na elaboração destes guias, uv não foi encontrado no PATH e não havia
.venv/Scripts/python.exe. Nenhuma dependência foi instalada nesta tarefa.

## 3. Preparar o histórico real

src/forecasting/dataset.py::load_demand_base exige o CSV da base tratada, com:

| Colunas | Uso |
|---|---|
| data_pedido | Data do registro e intervalo observado. |
| cd_setor, CICLOS | Identificadores carregados como texto. |
| nm_ciclo, aa_ciclo | Metadados obrigatórios no loader. |
| total_itens_mascarado | Alvo agregado. |
| total_pedidos_mascarado, total_volumes_mascarado | Predictores históricos opcionais, usados com defasagem. |
| Dt Abertura, Dt Fechamento | Janela observada. |
| dia_ciclo | Obrigatório no loader; não é feature direta de level. |

Qtde dias é recomposto quando ausente por dataset.py::load_demand_base.
demanda_level.csv do processamento do workbook tem outro esquema e não é entrada
direta desse loader.

dataset.py::_drop_fanned_out_rows elimina cópias de registros sob aberturas diferentes.
Verificar se a regra é apropriada à sua fonte.

~~~python
from src.forecasting.dataset import build_item_panel, cycle_calendar, load_demand_base

raw = load_demand_base("data/base_tratada_v2.csv")
calendar = cycle_calendar(raw)
history = build_item_panel(raw, calendar)
if history.empty:
    raise ValueError("Nenhum setor-ciclo completo encontrado.")
print(calendar.to_string(index=False))
print(history.head().to_string(index=False))
print(history.groupby("cd_setor").size().describe())
~~~

O painel produzido por dataset.py::build_item_panel contém:

| Campo | Definição |
|---|---|
| items, orders, volumes | Totais observados do setor-ciclo. |
| opening_date | Menor abertura registrada entre todos os setores do ciclo, por cycle_calendar. |
| window_start | Menor abertura do próprio setor-ciclo, por _sector_windows. |
| cycle_days | Fechamento menos abertura do setor, em dias corridos, mais 1. |

dataset.py::cycle_calendar descarta ciclos fora do intervalo observado de data_pedido.
Isso não comprova ausência de falhas internas na exportação.

Conferir totais não negativos, datas válidas, durações positivas, unicidade de
setor-ciclo, ciclos descartados e número de observações por setor.

**Ausência de registro não é zero.** dataset.py::build_item_panel não completa a grade.
features.py::_add_lags usa as linhas anteriores disponíveis do setor: com um ciclo
ausente, lag_1 pode não ser o ciclo global imediatamente anterior.

## 4. Escolher modelo e parâmetros

Primeiro executar o backtest do segundo guia e congelar a configuração escolhida.
Não selecionar um modelo apenas por seu nome.

Exemplos iniciais aceitos por src/config/schema.py:

~~~python
configs = {
    "naive": {"model": "naive", "strategy": "mean"},
    "ets_setor": {"model": "ets"},
    "ets_compartilhado": {"model": "ets", "alpha": "pooled"},
    "arima": {"model": "arima", "order": [1, 0, 0], "trend": "c"},
    "gbm": {
        "model": "lightgbm",
        "label": "gbm_referencia",
        "lags": [1, 2, 3, 4],
        "rolling_windows": [2, 3],
        "companion_lags": [],
        "target": "level",
        "num_leaves": 7,
        "n_estimators": 100,
        "learning_rate": 0.05,
    },
}
~~~

São configurações ilustrativas, não escolhas comprovadamente ótimas.

| Modelo | Parametrização e cuidado |
|---|---|
| ETS por setor | candidates/ets.py::build_ets usa ETS(A,N,N) por padrão. _minimum_observations exige 4 ciclos, 6 com tendência e 8 com tendência amortecida; sazonalidade exige também duas temporadas. |
| ETS compartilhado | schema.py::ETSParams permite alpha compartilhado só em ETS(A,N,N). _SharedAlphaCandidate._fit escolhe alpha e expoente da duração pelo erro no treino. |
| ARIMA | candidates/arima.py::_minimum_observations exige d + D*m + 2*(p+q+P+Q) + 2 observações. (1,0,0) exige 4. trend: c inclui intercepto com d=0; com diferenciação, tem outro significado. |
| LightGBM | features.py::complete_feature_frame perde linhas com lags/médias obrigatórios ausentes. _split_for_early_stopping valida dentro do treino quando configurado. |
| Sazonal naive | schema.py::NaiveParams exige season_length positivo. Confirmar a periodicidade do negócio e histórico suficiente. |

Mínimos técnicos não garantem qualidade. Para ETS com expoente fixo, ETSParams exige
alpha compartilhado; o intervalo do expoente não é explicitamente validado no schema.
Neste procedimento, uma escolha manual deve ficar em [0,1] e ser justificada no backtest.

TypeAdapter(ForecastParams) valida a configuração; campos desconhecidos podem ser
ignorados pelo padrão Pydantic. Conferir params.model_dump e o efeito real de cada campo.

## 5. Construir as consultas futuras

Essa construção ainda não está integrada em um executável da base real.
model.py::opening_scenarios e model.py::forecast já estão disponíveis, mas o chamador
precisa obter do calendário operacional:

1. código do ciclo futuro;
2. primeira abertura planejada do ciclo completo;
3. setores;
4. aberturas candidatas dos setores;
5. duração de cada janela ou fechamento para calculá-la.

No futuro, manter a mesma definição histórica de opening_date: primeira abertura do
ciclo inteiro. Não recalcular essa referência apenas nos setores consultados.

features.py::_calendar_features extrai o número do ciclo dos dois últimos caracteres
de CICLOS. Usar códigos consistentes e calendário real, sem supor duração constante
ou simplesmente incrementar o código atual.

Proposta: salvar data/forecast_inputs/level_queries.csv:

~~~csv
cd_setor,CICLOS,opening_date,window_start,cycle_days
A,202701,2027-01-04,2027-01-04,21
A,202701,2027-01-04,2027-01-09,16
B,202701,2027-01-04,2027-01-06,19
~~~

Substituir setores e datas pelos reais. O exemplo tem fechamento em 24/01/2027:

**cycle_days = (window_end - window_start).days + 1**

Não incluir os resultados efetivos futuros, como items/orders/volumes.

Alternativa: model.py::opening_scenarios(cycles, [0,5,10]) gera os mesmos offsets para
todos os setores-ciclos. Mantém a duração e desloca também o fechamento.
Para fechamento fixo ou opções por setor, montar as consultas diretamente.

## 6. Executar a previsão com a API atual

Proposta: salvar o exemplo como experiments/level_future_example.py.
Este guia não criou esse script nem os inputs.

~~~python
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

import src.forecasting.candidates  # registra os candidatos
from src.config.schema import ForecastParams
from src.forecasting.dataset import build_item_panel, load_demand_base
from src.forecasting.model import QUERY_KEYS, REGISTRY, forecast

history = build_item_panel(load_demand_base("data/base_tratada_v2.csv"))
if history.empty:
    raise ValueError("Histórico vazio.")

model_config = {"model": "ets", "alpha": "pooled"}  # trocar pela escolha validada
params = TypeAdapter(ForecastParams).validate_python(model_config)
candidate = REGISTRY.build(params)

queries = pd.read_csv(
    "data/forecast_inputs/level_queries.csv",
    dtype={"cd_setor": str, "CICLOS": str},
    parse_dates=["opening_date", "window_start"],
)
required = [*QUERY_KEYS, "opening_date"]
missing = set(required) - set(queries.columns)
if missing:
    raise ValueError(f"Colunas ausentes: {sorted(missing)}")
if queries.empty or queries[required].isna().any().any():
    raise ValueError("Consultas vazias ou com campos obrigatórios nulos.")

days = pd.to_numeric(queries["cycle_days"], errors="raise")
if not np.isfinite(days).all() or (days <= 0).any() or (days % 1 != 0).any():
    raise ValueError("cycle_days deve conter inteiros positivos.")
queries["cycle_days"] = days.astype(int)
if queries.duplicated(QUERY_KEYS).any():
    raise ValueError("Consultas duplicadas.")
if queries.groupby("CICLOS")["opening_date"].nunique().gt(1).any():
    raise ValueError("Referências inconsistentes no mesmo ciclo.")
if (queries["window_start"] < queries["opening_date"]).any():
    raise ValueError("Janela anterior à referência do ciclo.")
if queries["CICLOS"].isin(history["CICLOS"]).any():
    raise ValueError("Ciclo consultado já presente no histórico.")
if history["opening_date"].max() >= queries["opening_date"].min():
    raise ValueError("Consultas devem ser posteriores ao histórico.")

print("Configuração:", params.model_dump())
print("Setores sem histórico:", sorted(set(queries["cd_setor"]) - set(history["cd_setor"])))
answered = forecast(candidate, history, queries)
valid = answered["items_pred"].dropna()
if not np.isfinite(valid).all() or (valid < 0).any():
    raise ValueError("Previsão inválida.")
print(answered.to_string(index=False))
print("Sem previsão:", answered["items_pred"].isna().sum())

output = Path("results/forecasting/level_future.csv")
output.parent.mkdir(parents=True, exist_ok=True)
answered.to_csv(output, index=False)
~~~

Depois de criar o script e o CSV:

~~~powershell
uv run --extra forecast python -m experiments.level_future_example
~~~

model.py::forecast reajusta o candidato com todo o histórico fornecido.
Não precisa rodar backtest imediatamente antes; a configuração deve ter sido validada
anteriormente. O exemplo não carrega modelo serializado.

Para reproduzir decisões passadas, incluir só ciclos cujo resultado completo já estava
disponível na decisão. forecast verifica ordem das aberturas, mas não fechamento anterior
ou disponibilização dos dados. Conferir esses critérios na fonte operacional.

## 7. Verificar os outputs

model.py::forecast preserva consultas e adiciona items_pred por align_predictions.
Conferir chave, ordem, cobertura, valores finitos não negativos e plausibilidade
contra histórico e baseline.

| Setor | Ciclo | Abertura | Duração | items_pred ilustrativo |
|---|---|---|---:|---:|
| A | 202701 | 04/01/2027 | 21 | 500 |
| A | 202701 | 09/01/2027 | 16 | 460 |
| B | 202701 | 06/01/2027 | 19 | 120 |

Não somar os dois cenários de A: são alternativas. A unidade é itens.
Não substituir NaN automaticamente por zero.

model.py::opening_factors estima fatores por offset window_start - opening_date.
_OpeningAdjustedCandidate._factor usa 1 para offsets nunca vistos.
LightGBM usa features da janela em candidates/lightgbm.py::_scenario_rows.
Uma mudança de data pode resultar na mesma previsão em qualquer candidato.

## 8. Limites de vários ciclos futuros

Começar com horizonte 1 e validar no mesmo horizonte.

candidates/lightgbm.py::_forecast realimenta itens previstos, mas usa a média dos
cenários do setor-ciclo como lag do próximo, sem trajetórias independentes.
_pending_rows deixa pedidos, volumes e abertura futura em branco; _fill_predictions
preenche apenas items. Portanto, nos passos seguintes, features ligadas a essas
informações podem ficar ausentes.

model.py::_PerSectorCandidate.fit_predict associa a sequência de previsões aos
ciclos-alvo disponíveis do setor. Pular ciclos intermediários nas consultas pode
associar a previsão ao passo errado. Fornecer sequência coerente por setor.

Trajetórias completas por cenário exigem implementação adicional; uma tabela maior,
sozinha, não resolve essa necessidade.

## 9. Registrar a execução

Guardar configuração efetiva, versão do código, identificação/corte da base,
calendário futuro, consultas, CSV previsto e casos sem resposta.

Fluxo:
dataset.py → load_demand_base → dataset.py → build_item_panel
→ model.py → CandidateRegistry.build → model.py → forecast
→ candidato → fit_predict → model.py → align_predictions → chamador → CSV.

Nenhuma previsão da base real foi executada durante a elaboração deste documento.

