# Salvar uma pasta por execução do backtest

## 1. Escolha a pasta no YAML

O usuário define a pasta em `paths.run_dir`. Exemplo:

```yaml
split:
  min_train_cycles: 20
  horizon: 1
  window: expanding

candidates:
  - model: naive
    strategy: mean
  - model: ets
    alpha: pooled
    cycle_days_exponent: pooled

paths:
  base_csv: data/base_tratada_v2.csv
  run_dir: experiments/compare_forecasters/outputs/teste_ets_01
```

`experiments/compare_forecasters/records.py::_record_directory` interpreta
caminhos relativos a partir da pasta de onde o comando é executado. Execute
na raiz do repositório. `_create_directory` do mesmo arquivo exige uma pasta
nova: se ela existir, escolha outro nome. Isso preserva execuções anteriores.

## 2. Execute

Há um exemplo pronto em `configs/experiments/compare_forecasters/level_recorded.yaml`.
Edite seu `paths.run_dir` e execute, na raiz:

```bat
.\.venv\Scripts\python.exe -m experiments.compare_forecasters.run configs/experiments/compare_forecasters/level_recorded.yaml
```

`experiments/compare_forecasters/run.py::run` informa a pasta no terminal.
Esse exemplo inclui média, ETS compartilhado e LightGBM com região/gerência.

## 3. Consulte os arquivos

```text
teste_ets_01/
├── comparison.csv
├── predictions.csv
├── config.yaml
└── manifest.json
```

| Arquivo | Conteúdo | Arquivo e função responsável |
|---|---|---|
| `comparison.csv` | Métricas e cobertura dos modelos. | `run.py::_save_outputs`; `src/forecasting/comparison.py::compare` |
| `predictions.csv` | Valores reais, previsões e erros por setor, ciclo e rodada, com região (`CD_RE`), gerência (`CD_GV`), CD (`cd_cd`) e UF (`estado`) do setor-ciclo; CD e UF são os de mais itens quando há vários. | `run.py::_execute`, `_errors_by_candidate` e `_save_outputs`; `src/forecasting/dataset.py::sector_cycle_attributes` |
| `config.yaml` | Cópia exata do YAML lido para a execução. | `records.py::prepare_record` |
| `manifest.json` | Identificação da execução, parâmetros efetivos, modelos, features e versões. | `records.py::prepare_record` e `update_record` |

`records.py::prepare_record` registra:

- Identificador, horário com fuso `America/Sao_Paulo` e status inicial `running`.
- Caminho e SHA-256 da configuração e da base de dados.
- Histórico inicial, horizonte e janela, incluindo valores padrão do split.
- Nomes dos candidatos e seus parâmetros validados, com os padrões aplicados.
- Lista exata de features do LightGBM via `src/forecasting/features.py::feature_columns`.
- Para candidatos de série, lista de features tabulares vazia e `input_kind: item_series`;
  os inputs de janela são registrados em `scenario_inputs`.
- Commit Git, estado das alterações locais e versões de Python e bibliotecas.

`run.py::run` altera o status para `complete` após salvar os resultados, ou
`failed` com a mensagem de erro caso a execução falhe. Uma interrupção externa
pode deixar o status `running`; ele não comprova conclusão. Se a validação
inicial falhar antes da criação da pasta, não há registro de execução.

Os parâmetros efetivos são as configurações, incluindo padrões: por exemplo,
`lags: [1, 2, 3, 4]` se o campo for omitido no LightGBM. Parâmetros estimados em
cada rodada, como o alpha escolhido pelo ETS, e objetos treinados não são
exportados nesta versão. O estado Git informa alterações locais, mas não
arquiva uma cópia do código nem garante reprodução dessas alterações.

Para visualizar os runs, rode `uv run python dashboard/serve.py` na raiz: o
navegador abre o painel já com os runs desta pasta (ver `dashboard/README.md`).

## Compatibilidade com YAMLs anteriores

`run.py::_save_outputs` continua gravando nos caminhos `output_csv` e
`errors_csv` quando esses campos estiverem presentes. `records.py::_record_directory`
também cria uma pasta automática `run_<identificador>` dentro da pasta de
`output_csv`, com os quatro arquivos de registro e resultados.

Para adotar somente a pasta nomeada, mantenha `base_csv`, acrescente `run_dir`
e remova `output_csv` e `errors_csv`. Quando os três destinos são informados,
o arquivo de resultados é arquivado em `run_dir` e também exportado aos caminhos
antigos; estes últimos conservam o comportamento de sobrescrita anterior.
