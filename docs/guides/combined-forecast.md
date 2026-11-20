# Previsão diária: level × shape

Este fluxo conecta o level e o shape. A saída é por
**setor–ciclo–data de abertura–dia** no padrão atual:

```text
items_pred = level_pred × share_pred
```

O level prevê diretamente o total do setor; o shape prevê as participações dos
dias desse setor. Por exemplo, level de
1.200 e shares `[0,20; 0,30; 0,50]` produzem `[240; 360; 600]` itens diários.
O fluxo futuro nunca usa o total real do ciclo alvo. As quantidades permanecem
decimais para conservar o total previsto, sem arredondamentos independentes.

## 1. Escolha os modelos em um único YAML

Edite `configs/experiments/forecast_combined/future.yaml`. O padrão do level
é **ETS + distribuição compartilhada LightGBM até o setor**:

```yaml
history_cycles: 20
level:
  model: hierarchical
  target_level: sector
```

O ETS prevê a região e os LightGBMs compartilhados distribuem o total entre
gerências e setores: **região → gerência → setor**. Não há ajuste de modelo de
demanda por cidade nem soma de previsões de cidades para obter o level do setor.
O preset usa `history`, `calendar`, `structure`, `hierarchy` e `intermittency`.
`city_recent_shares` é omitido por padrão porque só possui valores no nível cidade.
Esses presets exigem `CD_RE` e `CD_GV` no histórico, além dos identificadores
de setor, estado e cidade; o último cadastro conhecido é usado na consulta.
As cidades são usadas para compor os totais históricos e as features de estrutura
do setor (quantidades de cidades, atividade e concentração). Essas features são
preservadas para manter a previsão do setor equivalente à etapa do modelo já
avaliado, sem prever a demanda das cidades.

Para reproduzir a saída antiga **setor-cidade**, configure explicitamente
`target_level: city` e `city_grouping: sector`. Para **gerência-cidade**, use
`target_level: city` e `city_grouping: management`. Nesse caso a sequência
é região → gerência → cidade e o bloco `history` é desabilitado por padrão.
As modalidades com cidade incluem `city_recent_shares`. Uma lista explícita em
`level.learning.features` substitui essa seleção automática. Os parâmetros
do LightGBM também podem ser alterados em `level.learning`.

O LightGBM usa 100 árvores, 7 folhas, taxa de aprendizado 0,05, mínimo de 20
registros por folha e semente 42. Nas modalidades com cidade, a regularização
de cidades usa 5 ciclos. As participações históricas usam `share_window: 5`.
O ETS escolhe alpha em `[0.05, 0.1, 0.2, 0.4, 0.8]` e expoente de duração
em `[0, 0.5, 1]`, usando apenas o treino. **20 ciclos de treino** e **5 ciclos
de participações** são parâmetros diferentes; a média móvel dos comparativos
também permanece fixada em cinco. Nenhuma busca entre modelos é feita no futuro.

### Resultados que motivaram os presets

Na arquitetura até setor-cidade, a etapa de **setor** do compartilhamento teve
MAE **1.850,85**, WMAPE **41,09%** e Bias **−6,52%**, nos resultados arquivados
de `2026-10-24_recent_city_shares_v01/active_city_analysis/comparison.csv`.
Essas são métricas por setor-ciclo, com todos os pontos desse nível; não são
métricas diárias nem métricas restritas a cidades ativas. A modalidade atual
encerra a distribuição nessa etapa e preserva suas features históricas.

Para executar um backtest isolado do level até setor (média móvel de cinco,
compartilhamento e reconciliação), use:

```bat
.venv\Scripts\python.exe -m experiments.hierarchical_level.run configs/experiments/hierarchical_level/sector.yaml
```

O forecast futuro usa somente compartilhamento; o YAML acima compara as
abordagens para validação, sem selecionar automaticamente uma vencedora.

Na conferência de 19/11/2026, os nove ciclos arquivados reproduziram as 5.506
previsões conhecidas de setor, com diferença máxima de 1,82e-12 item. Os três
pontos originalmente sem previsão mantêm o tratamento de zero do comparativo.

A validação diária de produção do ciclo 202612 está em
`experiments/forecast_combined/outputs/2026-11-19_sector_level_validation_v01/`.
Ela teve MAE de 259,99 itens, WMAPE de 122,19%, Bias de −4,96% e cobertura de
99,32% do volume real; 13 linhas reais não tiveram previsão disponível.
Esse resultado avalia a combinação com o shape, é exploratório em um único ciclo
e não demonstra boa acurácia diária. A equivalência do level não elimina esse erro.

As modalidades opcionais com cidade têm os resultados históricos abaixo.
As métricas abaixo são de level, condicionadas a cidades com demanda real
positiva, nos resultados históricos arquivados. Elas não são uma nova avaliação
da previsão diária nem uma comparação direta entre granularidades.

| Preset | MAE | WMAPE | Bias % | Pontos ativos |
| --- | ---: | ---: | ---: | ---: |
| Setor-cidade, todos os blocos | 220,31 | 57,04% | −21,72% | 43.951 |
| Gerência-cidade, sem `history` | 230,50 | 49,74% | −17,90% | 38.945 |

Fontes: `experiments/hierarchical_level/outputs/2026-10-24_recent_city_shares_v01/active_city_analysis/comparison.csv`
e `experiments/hierarchical_level/outputs/2026-10-27_management_city_ablation_v01/comparison.csv`.
Esses presets foram escolhidos pelo MAE entre as variantes de distribuição
compartilhada analisadas nesses recortes. Os resultados dependem do histórico
e não garantem o mesmo erro em novos ciclos.

### Como gerência-cidade continua alimentando o shape por setor

A saída diária continua identificando o setor. Para isso, o total previsto de
cada gerência–estado–cidade é repartido entre os setores planejados pela razão
entre suas compras históricas nos últimos `share_window` ciclos do treino.
Se todo o grupo não comprou nessa janela, usa-se o histórico completo do treino;
se também for zero, a repartição é uniforme entre os setores conhecidos.
Exemplo: compras de 30 e 10 dão participações de 75% e 25%.

Na modalidade `target_level: city`, `level_predictions.csv` registra `level_node`, `aggregate_level_pred`,
`sector_share` e `level_granularity`, além de `level_pred`. A soma de
`level_pred` preserva o total de cada nó agregado e a soma de `items_pred`
preserva o total de cada setor-cidade. No padrão `target_level: sector`, há uma
linha por setor-ciclo, com `level_node`, `level_pred` e `level_granularity=sector`,
sem colunas de cidade ou repartição entre cidades. O MAE de 230,50 avalia o total
gerência-cidade; **não avalia essa nova repartição aos setores**.

O histórico é limitado aos setores presentes na agenda da trajetória, para
que um subconjunto de setores não receba a demanda histórica de setores ausentes.
Para reproduzir o escopo regional completo, forneça a agenda completa.

O mesmo padrão é usado pelo `ForecastConfig` da aplicação e por
`configs/pipeline.yaml`, em `forecast.params`. No executor isolado, usa-se
`level`. Os modelos anteriores continuam aceitos em `src/config/schema.py`
(naive, ETS, ARIMA, LightGBM, cycle_factor). O shape continua configurável como
`uniform`, `sector` ou `lightgbm`; seu padrão permanece `sector`.

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

O cadastro de cidades conhecidas serve para construir as features de estrutura.
Na modalidade opcional com cidade, estado é mantido na identificação para não
misturar cidades homônimas. Setores sem
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

No preset hierárquico, cada etapa usa os últimos `history_cycles` ciclos
encerrados antes da primeira abertura daquela etapa. Os ciclos projetados
só entram no histórico de etapas posteriores quando sua janela inteira já
terminou; seus volumes são os previstos, nunca os reais futuros. A informação
disponível fica congelada em `as_of`. Etapas intermediárias podem exigir
histórico anterior à janela inicialmente selecionada; se não houver a
quantidade necessária de ciclos fechados, o executor informa o erro.

No padrão até setor, o histórico recursivo contém apenas nós de região,
gerência e setor. Não são fabricadas previsões por cidade. As características
de composição das cidades de um setor são mantidas no último estado conhecido
dos ciclos fechados de treino; atividade e concentração entre setores e
gerências são atualizadas a partir dos volumes previstos de seus filhos.

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
| `level_predictions.csv` | Total previsto por cenário–setor–ciclo; cidade somente em `target_level: city` |
| `shape_predictions.csv` | Participações por cenário–setor–ciclo–dia |
| `daily_predictions.csv` | Resultado final, incluindo abertura, data, dia relativo, level e share |
| `config_snapshot.yaml` | Configuração utilizada |
| `windows_snapshot.csv` | Agenda fornecida |
| `run_manifest.json` | Parâmetros efetivos, features, ciclos de treino, hashes, versões, revisão Git e status |

Cada execução exige uma pasta nova, para não sobrescrever resultados.
Verifique o status e a contagem `n_missing_levels` no manifest. O pipeline
verifica shares não negativos somando 1 e conservação do level por setor
(ou por cidade na modalidade opcional).
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
| `validation_by_sector.csv` | Métricas por setor–ciclo no padrão até setor |
| `validation_by_city.csv` | Métricas por setor–estado–cidade–ciclo, somente em `target_level: city` |
| `validation_windows.csv` | Janelas reconstruídas, etapas intermediárias e datas de corte |

Também são salvos o YAML e o manifest. O MAE diário em itens é a referência
principal; são reportados RMSE, WMAPE (razão, 0,25 = 25%), Bias e Bias %.
O sinal do erro é previsto menos real. As métricas são calculadas sobre os
pontos com previsão disponível: sempre leia também `n_unpredicted_rows`,
`unpredicted_actual_items` e `actual_volume_coverage`. Não compare métricas
com coberturas diferentes sem considerar a demanda que ficou de fora.
No padrão por setor, os itens reais de todas as cidades são somados por
setor–ciclo–dia **antes** de calcular MAE, WMAPE e Bias. O arquivo de validação
não usa cidades fictícias nem repete o total do setor para cada cidade.

## Código e fluxo

```text
experiments/forecast_combined/run.py → run
→ combined_inputs.py → prepare_histories / prepare_windows
→ combined.py → forecast_combined
→ hierarchy_future.py → forecast_hierarchical_path [level padrão]
  → hierarchy.py → forecast_distribution
  ou model.py → forecast                          [level legado]
→ shape_daily.py → predict_shape
  ou shape_learning.py → predict_learned_shape [shares previstos]
→ combined.py → combine_predictions
→ daily_predictions.csv
```

`combined_validation.py → validate_combined` reutiliza esse fluxo e acrescenta
a comparação histórica. Os modelos de level, o shape e os backtests anteriores
não são substituídos por este executor. `build_shape_days` recebeu apenas um
argumento opcional de calendário para manter o corte consistente entre painéis.
