# Manual simples: testar features da previsão de demanda

**Objetivo:** comparar versões do LightGBM para descobrir quais features ajudam
a prever o total de itens por setor e ciclo. Este manual é para Windows.

## 1. Abra o projeto

Abra a pasta do repositório no VS Code. No terminal, escolha **Command Prompt
(cmd)**. Os comandos abaixo usam esse terminal.

Entre na pasta do projeto. Troque o caminho pelo local do repositório no seu computador:

```bat
cd /d "C:\Users\SEU_USUARIO\Documents\GitHub\capacitated-assignment-problem"
```

Você deve estar na pasta que contém `src`, `configs`, `data` e `pyproject.toml`.
Confira se a base foi fornecida em `data/base_tratada_v2.csv`.

## 2. Confira o Python

```bat
py -3.13 --version
```

Deve aparecer `Python 3.13.x`. Se não aparecer, instale o Python 3.13 pelo
[site oficial](https://www.python.org/downloads/windows/), usando o instalador
de 64 bits para um Windows de 64 bits. Marque a opção de instalar `pip` e,
se disponível, adicionar Python ao PATH. Depois reabra o terminal e repita o comando.

Não é necessário instalar extensões do VS Code para executar estes comandos.

## 3. Crie o ambiente virtual — somente na primeira vez

```bat
py -3.13 -m venv .venv
```

Se o projeto já tem uma `.venv` funcionando, pule esse comando. O ambiente
fica na pasta `.venv`, separado dos outros projetos.

## 4. Instale as dependências — somente na primeira vez

Copie esta linha inteira:

```bat
.\.venv\Scripts\python.exe -m pip install "numpy>=2.0.0,<2.3" "pandas>=2.3.3" "pydantic>=2.11.0" "pyyaml>=6.0,<7.0" "lightgbm>=4.5.0" "statsmodels>=0.15.0" tzdata
```

Essas são as bibliotecas necessárias para este fluxo de forecasting; não
constituem a instalação completa de todas as funcionalidades do repositório.
As versões de forecasting estão em `pyproject.toml`; `tzdata` fornece os fusos
usados em `experiments/compare_forecasters/records.py::_now`.

Confira a instalação:

```bat
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import pandas, numpy, yaml, pydantic, lightgbm, statsmodels; import experiments.compare_forecasters.run; print('Ambiente OK')"
```

O primeiro comando deve informar que não há conflitos. O segundo deve mostrar
`Ambiente OK`. Usaremos sempre `.venv\Scripts\python.exe`, sem precisar ativar
o ambiente no terminal.

## 5. Crie um único arquivo de configuração

Crie `configs/experiments/compare_forecasters/teste_features.yaml` e cole:

```yaml
split:
  min_train_cycles: 20
  horizon: 1
  window: expanding

candidates:
  - model: lightgbm
    label: lgbm_base
    n_estimators: 100
    learning_rate: 0.05
    num_leaves: 7
    lags: [1, 2, 3, 4]
    rolling_windows: [3, 6]
    companion_lags: []
    categorical_features: []

  - model: lightgbm
    label: lgbm_com_regiao
    n_estimators: 100
    learning_rate: 0.05
    num_leaves: 7
    lags: [1, 2, 3, 4]
    rolling_windows: [3, 6]
    companion_lags: []
    categorical_features: [CD_RE]

paths:
  base_csv: data/base_tratada_v2.csv
  run_dir: experiments/compare_forecasters/outputs/2026-10-07_regiao_v01
```

Salve o arquivo. Use espaços para a indentação, sem TAB.

O exemplo compara o mesmo LightGBM sem região e com região. A função
`_evaluate_all()` de `experiments/compare_forecasters/run.py` executa as duas versões.

| Campo | O que você decide |
|---|---|
| `min_train_cycles` | Quantidade inicial de ciclos de treinamento. |
| `horizon` | Quantos ciclos seguintes serão previstos em cada rodada. Comece com 1. |
| `window: expanding` | O histórico cresce a cada rodada. `sliding` mantém a quantidade indicada em `min_train_cycles`. |
| `label` | Nome único da versão do LightGBM na tabela de resultados. |
| `base_csv` | Caminho da base fornecida para os testes. |
| `run_dir` | Pasta nova onde essa execução será salva. |

`RollingOriginSplit` e `_training_cycles()` de `src/forecasting/evaluation.py`
definem o histórico e horizonte; `LightGBMParams` de `src/config/schema.py`
define as opções dos candidatos.

## 6. Escolha a feature que quer testar

Mude somente uma opção na versão experimental, mantendo a versão base.

| Experimento | Opção no YAML |
|---|---|
| Região | `categorical_features: [CD_RE]` |
| Gerência de vendas | `categorical_features: [CD_GV]` |
| Região e gerência juntas | `categorical_features: [CD_RE, CD_GV]` |
| Pedidos, volumes e itens por pedido anteriores | `companion_lags: [1, 2, 3, 4]` |
| Outra seleção de itens anteriores | Por exemplo, `lags: [1, 2, 3]` |
| Outras médias históricas | Por exemplo, `rolling_windows: [2, 3]` |

Essas features são selecionadas por `feature_columns()` e calculadas pelas
funções de `src/forecasting/features.py`. Consulte o
[catálogo completo de features](forecasting-level-features.md) antes de escolher.
Os lags são observações anteriores do setor, não necessariamente ciclos
consecutivos do calendário.

Para cada experimento, ajuste o `label` e dê um nome novo à pasta, por exemplo:

```yaml
run_dir: experiments/compare_forecasters/outputs/2026-10-07_gerencia_v01
```

`_create_directory()` de `experiments/compare_forecasters/records.py` não
sobrescreve uma pasta existente. Para repetir um teste, use `v02`, depois `v03`.

## 7. Execute o backtest

Na raiz do projeto:

```bat
.\.venv\Scripts\python.exe -m experiments.compare_forecasters.run configs/experiments/compare_forecasters/teste_features.yaml
```

Aguarde a tabela final e o retorno do terminal à linha de comando. A função
`run()` de `experiments/compare_forecasters/run.py` mostra os modelos em execução
e salva os resultados. Este é o teste de desempenho das features contra dados reais.

## 8. Abra os resultados

Na pasta escolhida em `run_dir`, procure:

| Arquivo | Para que serve |
|---|---|
| `comparison.csv` | Comparar as métricas das versões. |
| `predictions.csv` | Investigar cada previsão, valor real e erro. |
| `config.yaml` | Conferir a configuração que foi executada. |
| `manifest.json` | Conferir modelos, parâmetros, features e status da execução. |

`_save_outputs()` de `experiments/compare_forecasters/run.py` salva os CSVs;
`prepare_record()` e `update_record()` de `records.py` salvam os registros.
Confira se o manifest apresenta `"status": "complete"`.

No `comparison.csv`, observe:

- **`mae_common`: menor é melhor.** É a métrica principal do ranking de level.
- **`wmape_common`: menor é melhor.** `0,15` representa 15%.
- **`bias_pct_common`: mais perto de zero significa menor desvio líquido.**
  Positivo indica superestimação; negativo, subestimação. Não use sozinho para escolher.
- **`n_scored_common` e `n_missing`:** mostram a cobertura das previsões.

Essas colunas são calculadas por `_candidate_row()` e `compare()` de
`src/forecasting/comparison.py`. As métricas `_common` usam os mesmos pontos
para todos os modelos. Detalhes no [guia de métricas](forecasting-level-metrics.md).

Registre em uma planilha: nome do teste, alteração feita, MAE base, MAE da versão
experimental e conclusão. Se o MAE cair de 100 para 95, a redução é de 5%.
Uma melhora nesse backtest ainda precisa ser confirmada em outros ciclos;
não escolha uma feature apenas pelo resultado de uma tentativa.

## 9. Se precisar programar uma feature nova

Os passos anteriores testam features que já existem. Uma feature nova,
como promoção ou feriado na janela, exige construção no código; uma coluna
nova no CSV sozinha não entra no modelo, pois `_dataset()` de
`src/forecasting/candidates/lightgbm.py` usa a lista de `feature_columns()`.

Comece em `src/forecasting/features.py::build_features` e `feature_columns`;
verifique também a chegada do dado pelo painel em
`src/forecasting/dataset.py::build_item_panel` e sua disponibilidade na previsão
em `src/forecasting/candidates/lightgbm.py::_forecast` e `_scenario_rows`.
Use apenas dados disponíveis quando a previsão é feita: o valor real do ciclo
previsto não pode entrar em uma feature daquele ciclo.

Antes de editar código, leia `AGENTS.md` na raiz. Para validar alterações,
instale as ferramentas uma vez:

```bat
.\.venv\Scripts\python.exe -m pip install uv ruff radon pytest pytest-cov
```

Depois execute, usando os arquivos que você alterou nos comandos de Ruff e Radon:

```bat
.\.venv\Scripts\uv.exe run --no-project --python .venv\Scripts\python.exe ruff format src/forecasting/features.py
.\.venv\Scripts\uv.exe run --no-project --python .venv\Scripts\python.exe ruff check src/forecasting/features.py
.\.venv\Scripts\uv.exe run --no-project --python .venv\Scripts\python.exe radon cc -s -a src/forecasting/features.py
.\.venv\Scripts\uv.exe run --no-project --python .venv\Scripts\python.exe pytest tests/forecasting/test_features.py tests/forecasting/test_dataset.py tests/forecasting/candidates/test_lightgbm.py -q --cov=src
```

Os testes devem passar, o Ruff não deve indicar erros e as funções devem ter
complexidade grau A. Acrescente testes que verifiquem o cálculo da feature
e que ela não utiliza a demanda real do alvo. Inclua também os testes de outros
módulos que você alterar. As convenções estão em `docs/conventions/testing.md`.

## Erros comuns

| Mensagem ou problema | O que fazer |
|---|---|
| Python 3.13 não encontrado | Instale a versão correta e reabra o terminal. |
| `.venv\Scripts\python.exe` não encontrado | Confira a pasta atual e execute o passo 3. |
| `ModuleNotFoundError` | Instale as dependências com o Python da `.venv`, conforme o passo 4. |
| Pasta da execução já existe | Mude `run_dir` para um nome novo. |
| Base não encontrada | Confira `base_csv` e a presença do CSV na pasta `data`. |
| Erro ao ler o YAML | Confira espaços, indentação e se o arquivo foi salvo em UTF-8. |

**Rotina após a primeira instalação:** editar o YAML → escolher uma pasta nova
→ executar o comando do passo 7 → consultar os resultados do passo 8.
