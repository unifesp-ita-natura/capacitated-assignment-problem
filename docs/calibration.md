# Calibração do Simulated Annealing contra o MIP

Este documento descreve o código da calibração dos parâmetros do SA
(`src/solver/heuristics/simulated_annealing.py`) usando o MIP de
`src/solver/mip/block_assignment.py`, resolvido com HiGHS, como referência de gap. O
experimento em si (pergunta, hipótese, desenho, resultado) está em
[`experiments/sa_calibration/README.md`](../experiments/sa_calibration/README.md).

## Regras de projeto

- **Nada é copiado.** Todo o domínio vem dos módulos originais: o gerador
  (`src/generate/generator.py`), o forecasting (`src/forecasting/`), os adaptadores de
  entrada (`src/solver/inputs.py`), o SA e o modelo do MIP. `src/calibration/` só
  orquestra essas funções e acrescenta o que não existia (gap, amostragem, agregação).
- **O único arquivo existente modificado é o SA**, com extensões retrocompatíveis
  (seção abaixo). MIP, `main.py`, `persistence`, `config`, `generate` e `forecasting`
  não foram tocados: o MIP é resolvido por uma `SolverFactory("appsi_highs")` criada
  dentro de `src/calibration/mip.py`, a config do experimento tem esquema próprio em
  `src/calibration/config.py` e o I/O fica em `src/calibration/io.py`.
- **`experiments/sa_calibration/run.py` só orquestra:** lê o YAML, chama
  `src/calibration`, grava os resultados e imprime o progresso.
- **Estilo:** `ruff check`/`ruff format` com a config do projeto; complexidade ciclomática
  ≤ 5 (nota A no `radon cc`) em todo código novo, sem `# noqa`.

## O que mudou no SA

| Mudança | Por quê | Impacto |
|---|---|---|
| `solve(..., callback=None)` e `_notify` | Instrumentar o traço (energia, T) sem duplicar o laço do SA | Sem callback, comportamento idêntico |
| `AnnealingResult.zmax`, `zmin`, `std_load` | Decompor o objetivo: `zmax − zmin = objective` (médias entre ciclos) | Campos novos com default, no fim |
| `AnnealingResult.time` e a propriedade `feasible` | Comparar execuções e marcar inviáveis (gap 100) | Idem |
| `_per_cycle_mean` (e `_balance_objective` passa a usá-lo) | A mesma iteração por ciclo servia ao objetivo e às métricas novas | `_balance_objective` dá o mesmo valor (coberto pelo teste existente) |
| `cooling_rate_for_budget(budget, T0, T_min)` | Separar orçamento de iterações do α | Função nova |
| `iterations_for_schedule(T0, T_min, α)` | Prever quantas iterações uma configuração gasta | Função nova |

A parada por estagnação não precisou de mudança: com `stagnation_tolerance = 0.0` a
condição `std < 0` nunca é verdadeira, e com `stagnation_window = max_iterations` a
janela nem enche. `resolve_params` aplica os dois em toda execução.

## Arquivos

### `src/calibration/`

| Arquivo | Papel |
|---|---|
| `config.py` | Esquema pydantic do YAML (`SACalibrationConfig`) e `load_calibration_config`. |
| `instances.py` | Monta uma instância a partir do gerador, do forecasting e de `solver.inputs`. |
| `mip.py` | Resolve o MIP com HiGHS e extrai status, limites, gap residual e solução; bissecção do aperto. |
| `gap.py` | Referência de gap (ótimo x limite inferior) e o gap de uma execução. |
| `space.py` | Espaço de busca: hipercubo latino (scipy) e TPE (Optuna). |
| `runner.py` | Uma execução do SA por (configuração, instância, seed), em paralelo, com traço. |
| `analysis.py` | Gap por execução, tabela por configuração, efeitos principais, regra de decisão. |
| `io.py` | CSV de execuções e traços, JSON do MIP, tabelas CSV/Markdown. |
| `plots.py` | PNG de convergência e de efeitos principais (matplotlib, extra `viz`). |

### Experimento e testes

| Arquivo | Papel |
|---|---|
| `experiments/sa_calibration/run.py` | Subcomandos `tighten`, `mip`, `screen`, `tpe`, `validate`, `extras`, `report`, `all`. |
| `experiments/sa_calibration/README.md` | Goal, hypothesis, design note, status e Result. |
| `experiments/sa_calibration/REPORT.md` | Relatório final: tabelas, gráficos, recomendação, limitações. |
| `experiments/sa_calibration/figures/` | Os PNGs citados no relatório (cópias de `outputs/report/`). |
| `experiments/sa_calibration/.gitignore` | Mantém `outputs/` fora do git. |
| `configs/experiments/sa_calibration/full.yaml` | A rodada real. |
| `configs/experiments/sa_calibration/smoke.yaml` | O mesmo pipeline em ~1 min, só para checar que funciona. |
| `tests/experiments/sa_calibration/` | Testes (o solver do MIP é sempre mockado). |

## Funções

### `config.py`

- `ExperimentDescription`, `InstanceSpec`, `Dimension`, `MipSettings`, `BudgetSettings`,
  `ScreenPhase`, `TpePhase`, `ValidatePhase`, `ExtrasPhase`, `TightenSettings`,
  `ReportSettings` — os blocos do YAML. `Dimension` rejeita intervalo vazio ou log com
  limite ≤ 0.
- `SACalibrationConfig` — o YAML inteiro; rejeita fase que cite instância não declarada.
- `load_calibration_config(path)` — lê e valida o YAML.

### `instances.py`

- `ForecastChoice` — estratégias de nível/forma pedidas no YAML (`None` = automática).
- `SyntheticHistory` — o que o gerador devolve para uma instância.
- `CalibrationInstance` — os dicionários de entrada do SA e do MIP.
  `sa_kwargs()`/`mip_kwargs()` devolvem os argumentos nomeados de `solve` e de
  `build_block_assignment_model`; `metadata()` resume a instância e as estratégias usadas.
- `synthesize_history(spec)` — `build_sectors`, `generate_synthetic_demand` e
  `build_sector_cd_assignment`, com o rng consumido na mesma ordem de `src/main.py` (a
  mesma seed gera a mesma instância que o pipeline principal).
- `resolve_strategies(history, spec, choice)` — completa a escolha do YAML com
  `score_and_select_strategies`.
- `forecast_calendar(history, spec, strategies)` — `forecast_future_cycles` +
  `forecast_items`.
- `build_instance(name, spec, choice)` — as três anteriores + `_to_solver_inputs`, que
  usa `build_id_maps`, `build_projected_demand`, `build_cd_sectors`,
  `build_daily_capacity`, `build_current_assignment_sa/_mip` e `group_days_by_cycle`.

> A **sequência** de chamadas repete a de `src/main.py` (é o mesmo pipeline), mas
> nenhuma função foi copiada. A alternativa — extrair uma função de `main.py` e
> reutilizá-la — exigiria modificar `main.py`, que está fora do escopo.

### `mip.py`

- `MipOutcome` — status (`termination`), `objective`, `lower_bound`, `upper_bound`, `gap`
  residual, `time`, `x`; `closed`, `has_incumbent`, `to_dict()`.
- `solve_mip(instance, settings, time_limit)` — `build_block_assignment_model(**mip_kwargs)`
  + `SolverFactory("appsi_highs").solve(model, timelimit=..., options={"mip_rel_gap": ...},
  load_solutions=False)`; carrega a solução só se houver incumbente (`_load_incumbent`).
- `finite_or_none`, `relative_gap`, `assignment_from_model` — auxiliares puros.
- `probe_feasible`, `min_feasible_multiplier` — bissecção do `capacity_multiplier`
  (subcomando `tighten`).

### `gap.py`

- `MipReference` — denominador do gap e de onde veio (`optimal`, `lower_bound`, `missing`).
- `mip_reference(record)` — incumbente se o MIP fechou, senão limite inferior.
- `compute_gap(objective, feasible, reference)` — `100·(obj − ref)/ref`; `100` se
  inviável; `NaN` se não há referência utilizável (fica fora das médias).

### `space.py`

- `scale_unit(dimension, u)` — leva u ∈ [0, 1] ao intervalo (log10 quando `log`).
- `lhs_configs(space, n, seed)` — hipercubo latino (`scipy.stats.qmc.LatinHypercube`).
- `distributions`, `create_study`, `warm_start`, `ask_batch` — TPE do Optuna: estudo
  multivariado com `constant_liar`, aquecido com todas as avaliações anteriores, pedindo
  um lote de configurações de uma vez.

### `runner.py`

- `RunSpec` — (fase, id da configuração, configuração, instância, seed); `key` identifica a
  execução para retomada.
- `TraceRecorder` — o callback do SA; guarda `(iteração, energia, melhor energia, T)` a
  cada `trace_every` iterações.
- `SARun` — spec, parâmetros efetivos, `AnnealingResult` e traço.
- `resolve_params(config, budget)` — `AnnealingParams(**config)` com estagnação desligada,
  `max_iterations` do orçamento e α derivado quando a configuração não fixa um.
- `run_sa(instance, spec, budget)` — uma execução com traço.
- `run_record(run)` — a linha do CSV (nomes do relatório: `p_cap`, `p_churn`, `zmax`...).
- `expand_specs(phase, configs, instances, seeds)` — o produto cartesiano.
- `run_many(instances, specs, budget, n_workers, on_result)` — `ProcessPoolExecutor`; as
  instâncias vão para cada processo uma vez só (`_init_worker`).

### `analysis.py`

- `add_gaps`, `add_group`, `prepare_runs` — gap, gap contra o incumbente e grupo
  (`tight`/`loose`) por execução.
- `summarize(runs, by)` — gap médio, desvio, P90, taxa de viabilidade, iterações, tempo.
- `config_scores`, `history_from_runs`, `configs_from_runs` — o histórico que aquece o TPE.
- `main_effects(runs, params)` — gap médio por quartil de cada parâmetro;
  `interval_label` formata as faixas.
- `always_feasible(runs)` — só as configurações viáveis em todas as execuções (separa o
  efeito de um parâmetro sobre a qualidade do efeito dele sobre a viabilidade);
  `feasibility_by(runs, param)` — taxa de viabilidade por faixa do parâmetro.
- `complexity(config)` — quantos parâmetros diferem do default do repositório.
- `rank_by_rule`, `ranked_summary`, `best_of` — a regra de decisão (menor gap médio →
  menor P90 → mais simples, com tolerância de empate em p.p.).
- `mip_table(records)` — a tabela das referências do MIP.

### `io.py` e `plots.py`

- `runs_path`, `trace_path`, `mip_path` — onde cada artefato mora.
- `append_run`, `read_runs`, `done_keys` — CSV incremental por fase (permite retomar).
- `write_trace`, `read_trace`, `write_json`, `read_json`, `write_table`, `to_markdown`.
- `plot_convergence` — melhor energia (painel de cima, recortada perto do valor final por
  `_energy_window`, porque o As-Is inviável começa ordens de grandeza acima) e temperatura
  em escala log (painel de baixo); `plot_main_effects` — um painel por parâmetro.

## Fluxo de dados

```text
full.yaml ──load_calibration_config──► SACalibrationConfig
                                            │
       InstanceSpec ─► synthesize_history (generator) ─► resolve_strategies (forecasting)
                                            │
                          forecast_calendar ─► _to_solver_inputs (solver.inputs)
                                            ▼
                                   CalibrationInstance
                  ┌─────────────────────────┴──────────────────────────┐
     mip_kwargs() ▼                                        sa_kwargs() ▼
  build_block_assignment_model                      space: lhs_configs / TPE ─► configs
  + SolverFactory("appsi_highs")                    runner: expand_specs ─► RunSpec
            ▼                                       resolve_params ─► AnnealingParams
  mip_optima/{inst}.json                            solve(..., callback=TraceRecorder)
            │                                                  ▼
            │                              runs_{fase}.csv  +  sa_runs/{cfg}_{inst}_{seed}.csv
            └──────► gap.mip_reference ──► analysis.prepare_runs (gap, grupo)
                                                   ▼
                     summarize / main_effects / ranked_summary / best_of
                                                   ▼
                  report/*.csv|md, convergence.png, effects_*.png, recommendation.json
```

A fase 2 lê de volta as execuções das fases 1 e 2 (`history_from_runs`) para aquecer o
TPE a cada lote; a fase 3 usa `ranked_summary` das fases 1+2 para escolher os finalistas;
a varredura de T_min parte de `best_of` (revalidação, se já rodou).

## Símbolos

| Símbolo | Campo | Significado de negócio |
|---|---|---|
| z_max, z_min | `zmax`, `zmin` | Pico e vale da captação diária de itens, em média entre os ciclos do horizonte. |
| z_max − z_min | `objective` | Desnivelamento: quanto a demanda diária oscila dentro do ciclo. É o que o SA e o MIP minimizam. |
| σ | `std_load` | Desvio-padrão da captação diária (média entre ciclos): oscilação "típica", não só a extrema. |
| P_cap | `p_cap` | Itens acima da capacidade diária somados sobre CDs e dias. 0 = nenhum CD estoura. |
| P_churn | `p_churn` | 2 × (setores realocados − limite de churn), se positivo. 0 = dentro do limite de mudanças. |
| ρ | `penalty_coefficient` | Peso das penalidades: E = objetivo + ρ·(P_cap + P_churn). |
| E(X) | `energy` | Energia que o SA minimiza. |
| T0, T_min, α | `initial_temperature`, `min_temperature`, `cooling_rate` | Agenda geométrica: T_{k+1} = α·T_k. |
| β | `sector_bias` | Probabilidade de escolher um setor de CD em violação para mover. |
| γ | `destination_bias` | Probabilidade de mover para uma combinação com folga no CD. |
| gap | `gap` | 100·(objetivo do SA − referência do MIP)/referência; 100 se o SA termina inviável. |
| gap residual do MIP | `gap` em `mip_optima` | (UB − LB)/UB: quão longe o MIP ficou de provar o ótimo. |
