# Level: passo a passo de backtesting, features e comparação

**Revisão:** 2026-10-05.  
**Escopo:** total de itens por setor-ciclo-janela, base real. Sem shape, avaliação diária ou solver.

Veja também [Previsões futuras](forecasting-level-futuro.md) para ambiente, esquema da
base, parametrização dos modelos e construção de consultas.
Este guia apresenta exemplos; os scripts/YAMLs abaixo não foram criados nesta tarefa.

## 1. Fixar o objetivo do experimento

Definir antes de executar: base, corte temporal, horizonte, janela de treino,
referências, variantes e métrica principal.

evaluation.py::RollingOriginSplit controla:

| Campo | Significado |
|---|---|
| horizon | Número de ciclos futuros por origem. |
| min_train_cycles | Mínimo global de ciclos no treino. |
| window: expanding | Todos os ciclos anteriores. |
| window: sliding | Últimos min_train_cycles. |

evaluation.py::evaluate gera N - min_train_cycles - horizon + 1 origens, quando positivo.
30 ciclos, mínimo 20 e horizonte 1 geram 10 origens.
O mínimo global não garante a mesma quantidade de observações em todos os setores.

Usar o horizonte desejado para a previsão operacional.
Reservar os ciclos finais antes da seleção de features; ver seção 6.

**Não incluir daily no YAML.** experiments/compare_forecasters/run.py::run ativa
avaliação diária pela presença desse bloco. Sem ele, comparison.py::compare ordena
por mae_common de level.

## 2. Preparar a base e entender o que será avaliado

dataset.py::load_demand_base exige o esquema descrito no primeiro guia.
dataset.py::build_item_panel agrega itens, pedidos e volumes por setor-ciclo e
remove ciclos classificados como incompletos por cycle_calendar.

evaluation.py::_score_origin usa a abertura e duração que realmente ocorreram,
fornece consultas sem items ao candidato e mantém os alvos reais separados.

Limitações importantes:

- não são geradas aberturas alternativas no backtest;
- setor-ciclo sem registros não é criado automaticamente com demanda zero;
- setor-ciclo presente sem previsão é registrado em missing;
- com horizonte >1, o mesmo setor-ciclo pode ser previsto em origens diferentes;
- features.py::_add_lags usa observações anteriores disponíveis por setor;
  ausência de ciclo pode fazer o lag saltar um ciclo global.

Melhor erro no passado não comprova a precisão de mudar uma abertura que nunca ocorreu.

## 3. Configurar referências e uma variante por hipótese

Proposta: salvar configs/experiments/level_features/all.yaml:

~~~yaml
experiment_name: level_features
split:
  horizon: 1
  min_train_cycles: 6
  window: expanding

candidates:
  - model: naive
    strategy: mean
  - model: ets
    alpha: pooled
  - model: arima
    order: [1, 0, 0]
    trend: c
  - model: lightgbm
    label: gbm_base
    target: level
    lags: [1, 2, 3, 4]
    rolling_windows: [2, 3]
    companion_lags: []
    num_leaves: 7
    n_estimators: 100
    learning_rate: 0.05
  - model: lightgbm
    label: gbm_companion
    target: level
    lags: [1, 2, 3, 4]
    rolling_windows: [2, 3]
    companion_lags: [1, 2, 3, 4]
    num_leaves: 7
    n_estimators: 100
    learning_rate: 0.05

paths:
  base_csv: data/base_tratada_v2.csv
  output_csv: experiments/level_features/outputs/comparison.csv
  errors_csv: experiments/level_features/outputs/errors_by_candidate.csv
~~~

Ajustar caminhos e min_train_cycles à base. Os parâmetros são exemplos iniciais,
não uma escolha comprovadamente ótima.

Só companion_lags muda entre os GBMs. features.py::companion_columns inclui juntos
pedidos, volumes e itens por pedido defasados: o experimento não isola cada variável.
schema.py::LightGBMParams.label distingue variantes.

Manter hiperparâmetros, alvo e split fixos ao avaliar uma família de features.
Testar target: ratio separadamente: candidates/lightgbm.py::_target_of muda o alvo
para items / level_ref, não acrescenta uma feature. features.py::_add_level_reference
calcula a média de ciclos anteriores do setor.

## 4. Executar e ler os outputs

Depois de criar o YAML:

~~~powershell
uv run --extra forecast python -m experiments.compare_forecasters.run configs/experiments/level_features/all.yaml
~~~

Para usar um YAML já existente:

~~~powershell
uv run --extra forecast python -m experiments.compare_forecasters.run configs/experiments/panel_order_counts/all.yaml
~~~

O segundo comando usa a base e outputs daquele arquivo e pode sobrescrever resultados.
Conferir caminhos antes de executar.

experiments/compare_forecasters/run.py::run carrega o painel, chama _evaluate_all,
cria candidatos por model.py::CandidateRegistry.build, chama evaluation.py::evaluate
e salva por _write:

| Arquivo/coluna | Conteúdo |
|---|---|
| comparison.csv | Ranking e cobertura por candidato. |
| errors_by_candidate.csv | Candidato, setor, ciclo, origem, actual, items_pred, abs_error. |
| mae_common | MAE por setor e média entre setores, no conjunto comum. |
| mae_own | Mesmo cálculo no conjunto próprio do candidato. |
| n_scored_common | Pontos comuns de setor-ciclo-origem. |
| n_scored_own | Pontos previstos pelo candidato. |
| n_missing | Alvos presentes sem previsão. |

comparison.py::common_scored_keys intersecta os pontos previstos por todos.
Se a interseção for vazia, mae_common é NaN; não há ranking válido.
Um candidato de pouca cobertura pode restringir a comparação de todos: avaliar também
GBM base e variante em um experimento apenas com esses dois.

O CSV de erros não exporta linhas missing nem todas as colunas de janela.
Usar EvaluationResult.missing pela API para inspecionar ausências. Para estudar
mudanças de abertura, juntar panel aos erros por cd_setor/CICLOS.

Exemplo ilustrativo:

**MAE 1.000 → 950 itens: melhoria = 100*(1.000-950)/1.000 = 5%.**

Conferir ganho por origem e por setor, cobertura e tamanho do conjunto comum.
Não aprovar somente por uma média ou por um limiar percentual universal.

## 5. Examinar métricas adicionais pela API

Bloco independente, para executar na raiz:

~~~python
import src.forecasting.candidates
from src.config.schema import ETSParams
from src.forecasting.dataset import build_item_panel, load_demand_base
from src.forecasting.evaluation import RollingOriginSplit, evaluate
from src.forecasting.model import REGISTRY

panel = build_item_panel(load_demand_base("data/base_tratada_v2.csv"))
result = evaluate(
    REGISTRY.build(ETSParams(alpha="pooled")),
    panel,
    RollingOriginSplit(horizon=1, min_train_cycles=6, window="expanding"),
)
print(result.summary())
print(result.missing.head().to_string(index=False))
print(result.scored.groupby("origin_cycle")["abs_error"].mean())
~~~

evaluation.py::EvaluationResult.summary informa MAE, RMSE, MASE, p90_abs_error,
worst_origin_mae e cobertura.

- evaluation.py::equal_weight_mae dá peso igual a cada setor.
- metrics.py::rmse agrega entre linhas e penaliza erros maiores.
- metrics.py::mase divide MAE por erro naive do treino;
  evaluation.py::_naive_in_sample_mae calcula o denominador apenas no treino.
- Denominador zero produz infinito; ausência de diferenças válidas produz NaN.
- EvaluationResult.summary lança erro se nenhum ponto foi avaliado.

Esses summaries usam os pontos próprios do candidato. Para comparar RMSE/p90 com
cobertura diferente, restringir ao conjunto comum primeiro. O exemplo abaixo recebe
results, uma lista retornada por várias chamadas a evaluate:

~~~python
import numpy as np
import pandas as pd

from src.forecasting.comparison import SCORE_KEYS, common_scored_keys
from src.forecasting.evaluation import equal_weight_mae
from src.forecasting.metrics import rmse

def common_level_metrics(results):
    keys = common_scored_keys(results)
    if not keys:
        raise ValueError("Sem pontos comuns.")
    common = pd.MultiIndex.from_tuples(sorted(keys), names=SCORE_KEYS)
    rows = []
    for result in results:
        scored = result.scored
        selected = pd.MultiIndex.from_frame(scored[SCORE_KEYS]).isin(common)
        scored = scored.loc[selected]
        rows.append({
            "candidate": result.candidate_name,
            "mae_common": equal_weight_mae(scored),
            "rmse_common": rmse(scored["actual"], scored["items_pred"]),
            "p90_common": float(np.percentile(scored["abs_error"], 90)),
            "n_common": len(scored),
        })
    return pd.DataFrame(rows).sort_values("mae_common")

# print(common_level_metrics(results).to_string(index=False))
~~~

MASE comum exige preservar o denominador de treino de cada origem.
Não recalcular o denominador com os alvos.

## 6. Separar desenvolvimento e holdout final

O driver padrão não reserva automaticamente um período final.
evaluation.py::evaluate percorre todas as origens elegíveis no painel recebido.

1. Reservar ciclos finais antes dos experimentos.
2. Selecionar features/configurações apenas nos ciclos anteriores.
3. Congelar a escolha.
4. Avaliar os ciclos reservados sem nova seleção.
5. Reajustar no histórico completo disponível para prever o futuro.

Exemplo, depois de construir panel:

~~~python
from pydantic import TypeAdapter

import src.forecasting.candidates
from src.config.schema import ForecastParams
from src.forecasting.evaluation import RollingOriginSplit, evaluate
from src.forecasting.model import REGISTRY

cycle_order = (
    panel[["CICLOS", "opening_date"]]
    .drop_duplicates()
    .sort_values("opening_date")["CICLOS"]
    .tolist()
)
H = 1
development_min_train = 6
if len(cycle_order) - H < development_min_train + H:
    raise ValueError("Poucos ciclos para desenvolvimento e holdout.")

development = panel[panel["CICLOS"].isin(cycle_order[:-H])]
chosen_config = {"model": "ets", "alpha": "pooled"}  # ilustrativo
params = TypeAdapter(ForecastParams).validate_python(chosen_config)

development_result = evaluate(
    REGISTRY.build(params),
    development,
    RollingOriginSplit(horizon=H, min_train_cycles=development_min_train),
)
print("Desenvolvimento:", development_result.summary())

# Executar só após congelar a escolha no desenvolvimento.
final_result = evaluate(
    REGISTRY.build(params),
    panel,
    RollingOriginSplit(horizon=H, min_train_cycles=len(cycle_order)-H),
)
print("Final:", final_result.summary())
~~~

min_train_cycles=N-H gera uma origem final com treino nos primeiros N-H ciclos
e alvo nos últimos H. Para um período final maior com várias origens, definir
explicitamente o protocolo em um driver adicional.

Referências:
[Validação temporal](https://otexts.com/fpp3/tscv.html) e
[Avaliação fora da amostra](https://otexts.com/fpp3/accuracy.html).

## 7. Testar features já disponíveis

Em schema.py::LightGBMParams e features.py::build_features:

| Opção | Hipótese |
|---|---|
| lags | Mais informação dos itens anteriores. |
| rolling_windows | Médias mais curtas/longas representam melhor o nível recente. |
| companion_lags | Pedidos, volumes e itens por pedido anteriores acrescentam sinal. |
| target: ratio | Separar tamanho do setor da correção prevista; é mudança de alvo. |

Configurações existentes:
configs/experiments/panel_order_counts/all.yaml,
configs/experiments/relative_target/all.yaml e
configs/experiments/lightgbm_short_window/all.yaml.

features.py::complete_feature_frame descarta linhas com features obrigatórias ausentes.
Lags maiores podem mudar o treino e a cobertura; medir ambos.
Manter labels únicos. Mudar uma família por vez, depois testar a combinação aprovada.

## 8. Acrescentar uma feature nova

Exemplo: dia da semana da abertura. É uma proposta, não implementada nesta tarefa.

1. Definir a variável a partir de window_start, não da abertura de referência.
2. Calcular em features.py::add_window_features.
3. Incluir em features.py::feature_columns.
4. Definir o tratamento de ausência em complete_feature_frame.
5. Para comparar com/sem, adicionar opção em schema.py::LightGBMParams e propagá-la
   por build_features, complete_feature_frame, feature_columns e métodos do candidato.
6. Conferir candidates/lightgbm.py::_scenario_rows: a variável deve ser recalculada
   depois de inserir a abertura consultada.
7. Testar treino e consultas futuras com o mesmo cálculo.
8. Rodar o experimento e comparar com a referência sem a feature.

Médias, desvio, tendência e contagens históricas devem usar só observações anteriores
do setor em features.py::build_features, com defasagem quando necessária.
Não agregar o painel completo antes de separar treino/teste.

Para informação externa conhecida antes da previsão, como promoção planejada:

- registrar a data de disponibilização;
- integrar atributos ao histórico;
- fornecer valores futuros nas consultas;
- ampliar evaluation.py::_score_origin, que atualmente seleciona apenas setor,
  ciclo, referência e colunas de janela;
- ampliar as colunas e propagação em candidates/lightgbm.py::_pending_rows e
  _scenario_rows. Coluna extra no CSV não entra automaticamente no modelo.

Não usar resultado real de evento futuro como predictor conhecido.

## 9. Verificar a implementação

Testes relevantes:
tests/forecasting/test_features.py,
tests/forecasting/test_scenarios.py e
tests/forecasting/candidates/test_lightgbm.py.

Cobrir cálculo manual, separação de setores, ausência de vazamento do alvo,
mudança de cenário, igualdade de entradas de treino/previsão, ausências e histórico curto.

Depois de uma futura alteração Python, conforme docs/conventions/testing.md:

~~~powershell
uv run --extra forecast ruff format src/forecasting src/config tests/forecasting
uv run --extra forecast ruff check src/forecasting src/config tests/forecasting
uv run --extra forecast radon cc -s -a src/forecasting src/config
uv run --extra forecast pytest tests/forecasting -q --cov=src
~~~

A convenção exige complexidade grade A e cobertura conforme risco.
Não é necessário executar solver para verificar forecasting.

## 10. Avaliar o ganho e distinguir modelos

Comparar MAE comum, RMSE/p90, erro por origem, cobertura, linhas de treino e holdout final.
Para features de abertura, examinar separadamente setores que mudaram de abertura;
comparison.py::compare não faz esse recorte automaticamente.

Melhor erro nas janelas observadas não prova efeito causal de mudar a abertura.
A conclusão também não cobre automaticamente setores sem histórico ou pares sem registros.

candidates/ets.py::build_ets e candidates/arima.py::_fit_and_forecast não consomem
features.py::build_features. Acrescentar colunas ali afeta apenas LightGBM quando
incluídas em suas entradas.

ARIMA com regressoras exige implementar exog no treino e na previsão SARIMAX;
a biblioteca suporta, mas o candidato atual não o usa:
[Documentação SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html).

ETSParams não fornece regressoras genéricas. ETS atual usa os ajustes externos de janela;
outras features exigiriam uma formulação/implementação adicional.

## 11. Guardar os resultados

Guardar base/corte, versão do código, configuração efetiva, protocolo, comparação comum,
cobertura, erros por origem/setor e casos sem previsão.

Fluxo:
experiments/compare_forecasters/run.py → run → dataset.py → build_item_panel
→ experiments/compare_forecasters/run.py → _evaluate_all
→ model.py → CandidateRegistry.build → evaluation.py → evaluate
→ evaluation.py → _score_origin → candidato → fit_predict
→ model.py → align_predictions → comparison.py → compare
→ experiments/compare_forecasters/run.py → _write.

Nenhum backtest completo da base real foi executado para elaborar este documento.

