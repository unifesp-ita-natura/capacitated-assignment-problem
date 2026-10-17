# Previsão diária: level × shape

Este fluxo conecta os modelos existentes sem mudar suas fórmulas nem os
backtests separados. A saída é por **setor–estado–cidade–ciclo–data de abertura–dia**:

```text
items_pred = level_pred × share_pred
```

O level prevê o total da cidade; o shape prevê as participações dos dias do
setor. A mesma curva é aplicada às cidades do setor. Por exemplo, level de
1.200 e shares `[0,20; 0,30; 0,50]` produzem `[240; 360; 600]` itens diários.
O fluxo futuro nunca usa o total real do ciclo alvo. As quantidades permanecem
decimais para conservar o total previsto, sem arredondamentos independentes.

## 1. Escolha os modelos em um único YAML

Edite `configs/experiments/forecast_combined/future.yaml`. O exemplo usa level
`naive:mean` e shape `sector`, como referência simples, sem afirmar que são os
melhores modelos. Você pode substituir o level pelos parâmetros aceitos em
`src/config/schema.py` (ETS, ARIMA, LightGBM, cycle_factor) e o shape por
`uniform`, `sector` ou `lightgbm`.

Exemplo de shape com features:

```yaml
shape:
  name: weekday
  model: lightgbm
  learning:
    features: [weekday]
    n_estimators: 100
    learning_rate: 0.05
    num_leaves: 15
    min_child_samples: 30
```

As outras features opcionais são `CD_RE`, `CD_GV` e `month`; consulte
[a referência](shape-feature-reference.md). Região e gerência são carregadas do
último cadastro conhecido no treino. Feriados não fazem parte deste fluxo.

## 2. Informe as janelas futuras

Preencha `configs/experiments/forecast_combined/future_windows.csv`. O arquivo
entregue contém apenas o cabeçalho: as aberturas reais de planejamento precisam
ser fornecidas pelo usuário, não inferidas dos pedidos futuros.

| Campo | Significado |
| --- | --- |
| `scenario_id` | Nome de uma trajetória completa de cenários, por exemplo `base` ou `abertura_antecipada` |
| `cd_setor` | Código de setor existente no histórico |
| `CICLOS` | Código de seis dígitos do ciclo, como nos dados históricos |
| `opening_date` | Primeira abertura geral do ciclo, comum a seus setores |
| `window_start` | Data de abertura da janela do setor nesse cenário |
| `window_end` | Último dia da janela do setor, inclusive |
| `emit` | `true` para exportar o ciclo; `false` para uma etapa intermediária de projeção |

Exemplo **ilustrativo**, a ser substituído por seu planejamento:

```csv
scenario_id,cd_setor,CICLOS,opening_date,window_start,window_end,emit
base,3257,202701,2027-01-05,2027-01-07,2027-01-16,true
later,3257,202701,2027-01-05,2027-01-09,2027-01-16,true
```

O primeiro cenário terá 10 dias; o segundo, 8. O código recalcula tanto o level
quanto o shape para cada cenário. Na saída, `window_start` é a data de abertura
do setor, `date` é o dia de calendário e `relative_day=1` é o primeiro dia da janela.

Todas as cidades conhecidas do setor são consultadas automaticamente. Estado
é mantido na identificação para não misturar cidades homônimas. Setores sem
histórico de cidades geram erro explícito. Uma cidade sem previsão de level
permanece com `level_status=missing` e valores vazios, sem substituir por zero.

### Ciclos intermediários e múltiplos ciclos

Cada `scenario_id` é uma trajetória independente. Para vários ciclos, liste
todas as etapas consecutivas desde o último ciclo disponível para treino;
isso preserva a previsão recursiva do LightGBM e o horizonte dos modelos
temporais. Cada ciclo de uma trajetória precisa listar os mesmos setores.

Se existe um ciclo intermediário já aberto mas ainda não encerrado no corte,
liste sua janela com `emit=false`. O modelo o projeta, sem usar seus pedidos,
para chegar ao ciclo solicitado. O executor avisa quando um ciclo conhecido
está faltando. Para ciclos futuros ainda não registrados na base, fornecer
a agenda completa continua sendo responsabilidade do planejamento.

## 3. Configure o corte do histórico

`as_of` é o instante local da previsão, sem sufixo de fuso, por exemplo
`2026-10-16T00:00:00`. Ajuste-o à execução real; ele precisa ser anterior à
primeira abertura geral de todos os ciclos com `emit=true`.

O treino utiliza apenas pedidos anteriores a esse instante e ciclos completos
com fechamento anterior ao dia do corte. `history_cycles: 20` seleciona os
últimos 20 ciclos globais elegíveis, não 20 observações obrigatórias de cada
cidade. Se não houver ciclos suficientes, a execução falha com uma explicação.

A regra de completude permanece a de `dataset.py → cycle_calendar`: a janela
inteira precisa caber na cobertura de datas da base observada. O conjunto de
cidades é construído antes de limitar aos últimos ciclos, preservando cidades
antigas e os zeros de ciclos sem compra previstos por `build_item_panel`.

## 4. Execute a previsão

Na raiz do repositório, com o ambiente virtual e dependências existentes:

```bat
.venv\Scripts\python.exe -m experiments.forecast_combined.run configs/experiments/forecast_combined/future.yaml
```

Não é necessário executar solver. Se escolher modelos que dependem de
LightGBM/statsmodels, use as dependências opcionais `forecast` do projeto.

Na pasta definida por `paths.run_dir`, serão gravados:

| Arquivo | Conteúdo |
| --- | --- |
| `level_predictions.csv` | Total previsto por cenário–setor–estado–cidade–ciclo |
| `shape_predictions.csv` | Participações por cenário–setor–ciclo–dia |
| `daily_predictions.csv` | Resultado final, incluindo abertura, data, dia relativo, level e share |
| `config_snapshot.yaml` | Configuração utilizada |
| `windows_snapshot.csv` | Agenda fornecida |
| `run_manifest.json` | Parâmetros efetivos, features, ciclos de treino, hashes, versões, revisão Git e status |

Cada execução exige uma pasta nova, para não sobrescrever resultados.
Verifique o status e a contagem `n_missing_levels` no manifest. O pipeline
verifica shares não negativos somando 1 e conservação do level por cidade.
Os dashboards existentes de level e shape continuam voltados a seus próprios
backtests; este novo formato não é carregado automaticamente por eles.

## 5. Valide o forecast diário com modelos fixos

Use `configs/experiments/forecast_combined/validation.yaml` para avaliar ciclos
passados. Escolha os mesmos blocos `level` e `shape` que deseja usar no futuro
e informe explicitamente `validation_cycles`. Não há lista de candidatos,
busca de hiperparâmetros ou seleção automática do melhor modelo. Cada modelo
continua estimando seus coeficientes normalmente no treino de cada origem.

```bat
.venv\Scripts\python.exe -m experiments.forecast_combined.run configs/experiments/forecast_combined/validation.yaml
```

Para cada ciclo, o executor fixa o corte um dia antes da primeira abertura,
prepara os históricos e chama a **mesma função de produção** `forecast_combined`.
Só depois junta a previsão com os pedidos reais. Dias previstos sem pedidos
entram com demanda real zero. Cidades novas, setores sem caminho completo de
janelas intermediárias e previsões ausentes ficam identificados como não
cobertos; o código não usa suas vendas para construir a lista de cidades futuras.

O replay utiliza as janelas efetivamente registradas na base, assumindo que
esse calendário seria conhecido no corte. A base não contém versões da agenda
nem datas de disponibilidade da informação; o replay não comprova essa
disponibilidade histórica. Não é possível validar causalmente aberturas
alternativas com o mesmo real observado. Ciclos incompletos são rejeitados.

Arquivos específicos da validação:

| Arquivo | Conteúdo |
| --- | --- |
| `daily_validation.csv` | Previsto e real por cenário observado–setor–estado–cidade–ciclo–dia |
| `validation_metrics.csv` | Métricas gerais e por ciclo |
| `validation_by_city.csv` | Métricas por setor–estado–cidade–ciclo |
| `validation_windows.csv` | Janelas reconstruídas, etapas intermediárias e datas de corte |

Também são salvos o YAML e o manifest. O MAE diário em itens é a referência
principal; são reportados RMSE, WMAPE (razão, 0,25 = 25%), Bias e Bias %.
O sinal do erro é previsto menos real. As métricas são calculadas sobre os
pontos com previsão disponível: sempre leia também `n_unpredicted_rows`,
`unpredicted_actual_items` e `actual_volume_coverage`. Não compare métricas
com coberturas diferentes sem considerar a demanda que ficou de fora.

## Código e fluxo

```text
experiments/forecast_combined/run.py → run
→ combined_inputs.py → prepare_histories / prepare_windows
→ combined.py → forecast_combined
→ model.py → forecast                         [level previsto]
→ shape_daily.py → predict_shape
  ou shape_learning.py → predict_learned_shape [shares previstos]
→ combined.py → combine_predictions
→ daily_predictions.csv
```

`combined_validation.py → validate_combined` reutiliza esse fluxo e acrescenta
a comparação histórica. Os modelos de level, o shape e os backtests anteriores
não são substituídos por este executor. `build_shape_days` recebeu apenas um
argumento opcional de calendário para manter o corte consistente entre painéis.
