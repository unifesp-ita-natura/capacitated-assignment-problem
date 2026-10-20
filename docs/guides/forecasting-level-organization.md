# Região e gerência como features de level

Consulte também o [catálogo de todas as features disponíveis](forecasting-level-features.md).

`src/config/schema.py::LightGBMParams` aceita `categorical_features` com
`CD_RE` (região), `CD_GV` (gerência de vendas), ambas ou nenhuma. A configuração
padrão é `[]`, preservando os experimentos existentes.

```yaml
- model: lightgbm
  label: lgbm_regiao_gerencia
  categorical_features: [CD_RE, CD_GV]
  n_estimators: 100
  learning_rate: 0.05
  num_leaves: 7
```

## Executar a comparação

Na raiz do repositório, usando o ambiente `.venv` já instalado:

```bat
.\.venv\Scripts\python.exe -m experiments.compare_forecasters.run configs/experiments/compare_forecasters/level_organization.yaml
```

Esse YAML compara quatro LightGBM com os mesmos parâmetros: sem organização,
somente região, somente gerência e ambas. Treina inicialmente em 20 ciclos,
usa janela crescente e horizonte de 1 ciclo. Não altera `level_v2.yaml`.

Resultados:

- `experiments/compare_forecasters/outputs/level_organization_comparison.csv`
- `experiments/compare_forecasters/outputs/level_organization_errors.csv`

`src/forecasting/comparison.py::compare` mantém o ranking de level por
`mae_common`, com as demais métricas no mesmo conjunto comum. Acrescentar
features não garante redução do erro; use os resultados para decidir.

## Como os códigos chegam ao modelo

1. `src/forecasting/dataset.py::load_demand_base` lê os códigos como texto,
   preservando zeros à esquerda.
2. `build_item_panel` e `_attach_organization` do mesmo arquivo mantêm o vínculo
   por setor-ciclo. Mais de um código não nulo dentro do mesmo setor-ciclo
   gera erro de validação; mudanças entre ciclos são permitidas. As colunas
   são opcionais na base, mas precisam existir no histórico se ativadas.
3. `src/forecasting/features.py::feature_columns` acrescenta somente os códigos
   solicitados. `complete_feature_frame` verifica a presença das colunas e
   permite códigos nulos, que LightGBM trata como valores ausentes.
4. `src/forecasting/candidates/lightgbm.py::_category_types` define as categorias
   pelo histórico de treinamento. `_as_model_frame` aplica a mesma codificação
   no treino, validação interna e previsão; `_dataset` declara os campos como
   categóricos, sem tratar o número do código como uma quantidade.
5. `_carry_organization` do mesmo arquivo usa o último código não nulo observado
   por setor no histórico disponível. O backtest não lê os códigos do ciclo
   avaliado. Previsões de vários ciclos carregam esse vínculo adiante.

Essa regra supõe manutenção do último vínculo conhecido na previsão futura.
Um cadastro planejado de mudanças futuras ainda não é recebido pelo candidato.
Na base `base_tratada_v2.csv` inspecionada em 2026-10-06, há 8 regiões e 37
gerências, sem nulos ou mudanças de vínculo entre os ciclos de cada setor.

As features são utilizadas pelo LightGBM. Os candidatos atuais ETS, ARIMA e
média histórica mantêm seu comportamento.
