# Referência de features disponíveis para previsão de level

**Referência do código em 06/10/2026.**

Este catálogo descreve as features implementadas para o **LightGBM de level**:
o modelo que prevê o total de itens de um setor em um ciclo. A lista que entra
no modelo é definida por `feature_columns()` de
[src/forecasting/features.py](../../src/forecasting/features.py) e selecionada por
`_PooledLightGBMCandidate._columns` e `_dataset()` de
[src/forecasting/candidates/lightgbm.py](../../src/forecasting/candidates/lightgbm.py).

## 1. Features automáticas

Estas features sempre entram no LightGBM atual; o YAML não tem opções
individuais para desligá-las. Essa seleção está em `feature_columns()` de
`src/forecasting/features.py`.

| Feature | Significado e origem | Tipo | Arquivo e função responsável |
|---|---|---|---|
| `cd_setor` | Identificador do setor. Permite distinguir setores. | Categórica | `lightgbm.py::_category_types`, `_as_model_frame` e `_dataset` |
| `cycle_number` | Últimos dois caracteres de `CICLOS`, convertidos para inteiro. Ex.: `202612` → `12`. | Numérica | `features.py::_calendar_features` |
| `opening_month` | Mês da primeira abertura do ciclo, comum aos setores. Ex.: fevereiro → `2`. | Numérica | `features.py::_calendar_features` |
| `cycle_days` | Duração da janela do setor, contando abertura e fechamento. Na previsão, vem da janela consultada. | Numérica, dias | `dataset.py::_sector_windows`; `lightgbm.py::_scenario_rows` |
| `days_since_previous_opening` | Diferença em dias entre a abertura da janela consultada e a abertura da observação anterior do setor. | Numérica, dias | `features.py::build_features` e `add_window_features`; `lightgbm.py::_scenario_rows` |
| `opening_day_of_month` | Dia do mês de abertura da janela do setor: 1 a 31. | Numérica | `features.py::add_window_features`; `lightgbm.py::_scenario_rows` |

O calendário geral vem de `cycle_calendar()` de
[src/forecasting/dataset.py](../../src/forecasting/dataset.py): `opening_date` é
a menor `Dt Abertura` do ciclo. A janela do setor vem de `_sector_windows()`
do mesmo arquivo: `window_start` é sua menor abertura e `cycle_days` é
`(maior fechamento − menor abertura) + 1`.

`_dataset()` de `src/forecasting/candidates/lightgbm.py` declara somente setor
e os códigos organizacionais selecionados como categóricos. Número do ciclo,
mês e dia do mês são enviados como valores numéricos.

## 2. Features configuráveis de histórico

Todos os históricos abaixo são construídos **dentro de cada setor**, em ordem
de `opening_date`, por `build_features()` de `src/forecasting/features.py`.

| Família de features | O que representa | Como ativar no YAML | Arquivo e função responsável |
|---|---|---|---|
| `lag_k` | Itens na k-ésima observação anterior do setor. Ex.: `lag_1`, `lag_2`. | `lags: [1, 2]` | `features.py::_add_lags` |
| `rolling_mean_w` | Média dos itens das w observações anteriores. Ex.: `rolling_mean_3`. Exclui o ciclo da própria linha. | `rolling_windows: [3]` | `features.py::_add_rolling_means` |
| `orders_lag_k` | Total de pedidos na k-ésima observação anterior. | `companion_lags: [1, 2]` | `features.py::_add_companion_lags` |
| `volumes_lag_k` | Total de volumes na k-ésima observação anterior. | `companion_lags: [1, 2]` | `features.py::_add_companion_lags` |
| `items_per_order_lag_k` | Itens / pedidos na k-ésima observação anterior. Divisões que resultam em infinito viram valor ausente. | `companion_lags: [1, 2]` | `features.py::_add_items_per_order_lags` |

`companion_lags` ativa as três famílias em conjunto. A configuração atual
não permite ativar pedidos, volumes e itens por pedido separadamente;
`companion_columns()` de `src/forecasting/features.py` lista as três.

**Exemplo numérico:** um setor tem itens `[100, 200, 300]` e pedidos `[10, 20, 15]`
nas três observações anteriores. Para a próxima observação:

```text
lag_1 = 300
lag_2 = 200
rolling_mean_3 = (100 + 200 + 300) / 3 = 200
orders_lag_1 = 15
items_per_order_lag_1 = 300 / 15 = 20
```

Esses cálculos correspondem a `_add_lags()`, `_add_rolling_means()`,
`_add_companion_lags()` e `_add_items_per_order_lags()` de
`src/forecasting/features.py`.

**Lag significa observação anterior disponível, e não necessariamente ciclo
anterior do calendário.** `build_item_panel()` de `src/forecasting/dataset.py`
não cria registros zero para combinações setor-ciclo ausentes; `_add_lags()`
de `src/forecasting/features.py` aplica `shift(k)` sobre os registros existentes.

## 3. Features opcionais de região e gerência

| Feature | Significado | Como ativar no YAML | Arquivo e função responsável |
|---|---|---|---|
| `CD_RE` | Código da região do setor. | `categorical_features: [CD_RE]` | `dataset.py::load_demand_base` e `_attach_organization`; `lightgbm.py::_category_types`, `_as_model_frame`, `_dataset` e `_carry_organization` |
| `CD_GV` | Código da gerência de vendas do setor. | `categorical_features: [CD_GV]` | Mesmos arquivos e funções da linha anterior |

Para ambas, use `categorical_features: [CD_RE, CD_GV]`.
`LightGBMParams` de [src/config/schema.py](../../src/config/schema.py) aceita
somente esses dois nomes nessa opção.

`load_demand_base()` de `src/forecasting/dataset.py` lê os códigos como texto,
preservando zeros à esquerda. `_attach_organization()` do mesmo arquivo mantém
um vínculo por setor-ciclo e rejeita mais de um código não nulo dentro dessa
combinação. Mudanças entre ciclos são permitidas.

No treinamento, `_dataset()` de `src/forecasting/candidates/lightgbm.py` usa
os vínculos históricos de cada linha. Na previsão e no backtest,
`_carry_organization()` do mesmo arquivo leva o último código não nulo observado
por setor no histórico da rodada; não utiliza o código do ciclo avaliado.
Um cadastro de mudanças organizacionais planejadas para ciclos futuros ainda
não é recebido pelo candidato.

## 4. Valores padrão e exemplo completo

Se omitidos, os campos de `LightGBMParams` em `src/config/schema.py` são:

```yaml
lags: [1, 2, 3, 4]
rolling_windows: [3, 6]
companion_lags: []
categorical_features: []
target: level
```

Pela lista de `feature_columns()` de `src/forecasting/features.py`, isso gera
**12 features**: 6 automáticas, 4 lags de itens e 2 médias móveis.

Para usar todas as famílias disponíveis, com quatro lags:

```yaml
candidates:
  - model: lightgbm
    label: lgbm_todas_as_familias
    target: level
    lags: [1, 2, 3, 4]
    rolling_windows: [3, 6]
    companion_lags: [1, 2, 3, 4]
    categorical_features: [CD_RE, CD_GV]
    n_estimators: 100
    learning_rate: 0.05
    num_leaves: 7
```

Esse trecho deve ficar no YAML do backtest, junto dos blocos `split` e `paths`.
`experiments/compare_forecasters/run.py::_evaluate_all` lê `candidates` e valida
os parâmetros com `ForecastParams` de `src/config/schema.py`.

O exemplo produz **26 features**, segundo `feature_columns()` de
`src/forecasting/features.py`: 12 do padrão, 12 de pedidos/volumes/itens por
pedido e 2 organizacionais. Outros lags e tamanhos de média podem ser escolhidos;
use inteiros positivos. Mantenha pelo menos um lag de itens: `_predictable_rows()`
de `src/forecasting/candidates/lightgbm.py` exige algum lag de itens disponível
para gerar previsão. Essa recomendação não é uma validação de positividade
implementada em `LightGBMParams`.

## 5. Histórico insuficiente e previsão de vários ciclos

`complete_feature_frame()` de `src/forecasting/features.py` remove do treino
linhas sem algum lag ou média exigido. Por exemplo, uma média de 6 observações
só fica disponível após 6 observações anteriores daquele setor. Aumentar os
lags pode reduzir o conjunto de treinamento.

A mesma função verifica a existência das colunas organizacionais solicitadas,
mas permite seus valores nulos e valores ausentes nas features de janela.
`_as_model_frame()` de `src/forecasting/candidates/lightgbm.py` transforma códigos
fora do vocabulário de treinamento em valores ausentes.

Para prever mais de um ciclo, `_forecast()` e `_fill_predictions()` de
`src/forecasting/candidates/lightgbm.py` reutilizam as previsões de itens nos
lags seguintes. Já `_pending_rows()` deixa pedidos e volumes futuros ausentes:
eles não são previstos por esse candidato. Assim, features companion podem
ficar ausentes nos passos seguintes.

Também em `_pending_rows()` de `src/forecasting/candidates/lightgbm.py`, a janela
da linha recursiva fica inicialmente vazia; `_scenario_rows()` aplica a janela
somente à consulta de previsão. Por isso, em passos posteriores ao primeiro,
`days_since_previous_opening` pode ficar ausente. Os vínculos organizacionais
são carregados adiante por `_carry_organization()`.

## 6. Colunas auxiliares, alvo e parâmetros do modelo

Nem toda coluna do painel é uma feature. `feature_columns()` de
`src/forecasting/features.py` define a seleção efetiva.

| Campo | Uso atual | Arquivo e função responsável |
|---|---|---|
| `items` | Demanda real e alvo quando `target: level`. | `dataset.py::build_item_panel`; `lightgbm.py::_target_of` |
| `orders`, `volumes` | Matéria-prima para lags; os valores do ciclo previsto não entram diretamente. | `features.py::_add_companion_lags`; `lightgbm.py::_pending_rows` |
| `CICLOS`, `opening_date`, `window_start` | Identificação, ordenação e cálculo das features. | `features.py::_calendar_features`, `build_features` e `add_window_features` |
| `previous_window_start` | Auxiliar para calcular o intervalo desde a abertura anterior. | `features.py::build_features` e `add_window_features` |
| `level_ref` | Média acumulada dos itens anteriores do setor. É referência de transformação do alvo, não uma feature de entrada. | `features.py::_add_level_reference`; `lightgbm.py::_target_of` e `_to_items` |
| `items_pred` | Saída da previsão em itens. | `lightgbm.py::_forecast` |

Com `target: ratio`, `_target_of()` de `src/forecasting/candidates/lightgbm.py`
treina sobre `items / level_ref`; `_to_items()` multiplica a razão prevista por
`level_ref` para voltar a itens. Isso muda o alvo, sem acrescentar uma feature.

`n_estimators`, `learning_rate`, `num_leaves`, `min_child_samples`, regularização,
amostragem e early stopping são parâmetros de treinamento definidos em
`LightGBMParams` de `src/config/schema.py` e usados por `_settings()` e `_fit()`
de `src/forecasting/candidates/lightgbm.py`; não são features.

## 7. Aplicação aos outros modelos e limites do catálogo

Os ETS e ARIMA atuais não recebem esta tabela de features. Seus métodos
`_fit_and_forecast()` em [ets.py](../../src/forecasting/candidates/ets.py) e
[arima.py](../../src/forecasting/candidates/arima.py) recebem a série de itens
do setor. As adaptações de duração e abertura ficam em `window_scaled()` e
`opening_adjusted()` de [model.py](../../src/forecasting/model.py); o ETS
compartilhado também trata duração em `_SharedAlphaCandidate` de `ets.py`.

Campos da base como feriado, cidade, estado, rota e dia da semana não estão
incluídos na seleção atual de `feature_columns()` de `src/forecasting/features.py`.
Adicionar uma coluna ao CSV não a ativa automaticamente no modelo.

## Referências para execução

- [Comparação com região e gerência](forecasting-level-organization.md)
- [Métricas do backtest](forecasting-level-metrics.md)
- [Configuração pronta para comparar organização](../../configs/experiments/compare_forecasters/level_organization.yaml)

Ao acrescentar ou remover features em `src/forecasting/features.py::feature_columns`,
atualize este catálogo para acompanhar o código.
