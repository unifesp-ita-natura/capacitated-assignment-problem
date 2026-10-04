# sa_calibration

**Name:** `sa_calibration` — driver `experiments/sa_calibration/run.py`, configs
`configs/experiments/sa_calibration/{full,smoke}.yaml`, código em `src/calibration/`,
saídas em `experiments/sa_calibration/outputs/` (fora do git).

**Status:** `confirmed` — rodada completa em 2026-09-28; ver **Result** e [`REPORT.md`](REPORT.md).

## Goal

Quais valores de `initial_temperature` (T0), `min_temperature` (T_min), `sector_bias`
(β), `destination_bias` (γ) e `penalty_coefficient` (ρ) minimizam o gap percentual do
Simulated Annealing contra o ótimo do MIP resolvido pelo HiGHS, com orçamento fixo de
200 mil iterações?

## Hypothesis

1. No relatório anterior (gap contra o BKV do próprio SA) o `cooling_rate` dominou
   porque controlava **quantas iterações** a busca tinha. Com α derivado do orçamento
   (`cooling_rate_for_budget`), toda configuração gasta as 200 mil iterações e o que
   sobra é o efeito real da agenda de temperatura.
2. T_min = 0.001 desperdiça o fim da busca (o relatório anterior viu a melhor energia
   parar em T ≈ 12). Esperamos T_min na faixa 1–10 melhor que 0.001.
3. β e γ só importam quando há CD em violação (instâncias `tight*`); nas folgadas
   (`base*`) o efeito deles deve ser ruído. ρ importa nas duas por causa do churn.
4. ρ = 1e6 (default) torna qualquer violação inaceitável em qualquer temperatura, o que
   impede atravessar regiões inviáveis; um ρ intermediário deve ajudar nas apertadas.

## Design note

- **Referência:** MIP de `src/solver/mip/block_assignment.py` resolvido com
  `appsi_highs`, `mip_rel_gap = 0.01`, limite de 600 s (40 setores) / 1800 s (80).
  `gap = 100·(obj_SA − ref)/ref` se a execução do SA termina viável, `100` se inviável;
  `ref` é o incumbente do MIP se ele fechou e o limite inferior se não fechou. O gap
  residual do MIP é sempre reportado (`mip_reference.csv`).
- **Instâncias** (gerador sintético + forecasting do repo):

  | nome | setores | ciclos no horizonte | `capacity_multiplier` | papel |
  |---|---|---|---|---|
  | `base40` | 40 | 2 | 1.0 | folgada (carga ≤ 0,4% da capacidade) |
  | `tight40` | 40 | 2 | 0.0027 (1,15 × mínimo viável) | apertada; As-Is estoura um CD em ~48% |
  | `base80` | 80 | 3 | 1.0 | folgada, maior |
  | `tight80` | 80 | 3 | 0.0072 (1,03 × mínimo viável) | apertada, **nunca vista** até a fase 3 |

  O mínimo viável vem de `run tighten` (bissecção com o MIP). Em 80 setores o ótimo do
  MIP não muda com o multiplicador (o churn é que prende); o aperto de `tight80` é o As-Is
  começar ~5% acima da capacidade, o que obriga o SA a reparar a solução.
- **Fixos em toda execução:** `max_iterations = 200_000`, parada por estagnação
  desligada (`stagnation_window = max_iterations`, `stagnation_tolerance = 0.0`) e α
  derivado de (T0, T_min) — exceto nas configurações nomeadas que fixam α (`default_repo`,
  `prev_*`), mantidas assim de propósito para comparação.
- **Fases:**
  1. `screen` — 60 configurações em hipercubo latino × seeds 1–5 × `base40`, `tight40`,
     `base80`.
  2. `tpe` — 3 lotes × 12 configurações do TPE do Optuna (multivariado, `constant_liar`),
     aquecido com todas as avaliações anteriores, × seeds 1–5 × `base40`, `tight40`.
  3. `validate` — 4 finalistas (regra de decisão sobre as fases 1+2) + `default_repo` e
     `default_budget`, × seeds 101–108 × `base40`, `tight40`, `tight80`.
  4. `extras` — varredura T_min ∈ {0.001, 0.01, 0.1, 1, 5, 10} na configuração
     recomendada, e as configurações do relatório anterior (`prev_*`, `cal_long`),
     × seeds 101–105 × `base40`, `tight40`.
- **Regra de decisão:** menor gap médio; empate (±0,5 p.p.) → menor P90 do gap; empate →
  a mais simples (menos parâmetros diferentes do default do repositório).
- **Limitações conhecidas:** instâncias sintéticas e pequenas (40/80 setores; o
  problema real tem ~800); uma única instância por tamanho/aperto (a variação entre
  instâncias não é medida); `statsmodels` ausente no ambiente muda a estratégia de
  previsão escolhida automaticamente (Holt/ARIMA são pulados) — a escolha efetiva fica
  gravada em `mip_optima/*.json`; os esquemas de resfriamento (b) patamares e (c)
  adaptativo por taxa de aceitação não foram testados porque exigiriam um gancho de agenda
  no SA, fora do escopo combinado.

## Como rodar

O projeto não tem extra `calibration` (o `pyproject.toml` é de outros mantenedores).
Instalar o Optuna com `uv pip install` seria desfeito no próximo `uv sync`, então rode com
`--with`:

```bash
RUN="uv run --extra viz --with optuna python -m experiments.sa_calibration.run"
$RUN tighten   # opcional: recalcula o mínimo viável das instâncias apertadas (vai para o YAML)
$RUN mip       # mip_optima/{instância}.json
$RUN screen    # runs_screen.csv + sa_runs/*.csv
$RUN tpe       # runs_tpe.csv, tpe_batches.json
$RUN validate  # runs_validate.csv, finalists.json
$RUN extras    # runs_extras.csv
$RUN report    # report/*.csv|md|png, recommendation.json
# ou tudo em sequência (menos o tighten):
$RUN all
```

Cada fase retoma de onde parou: execuções já gravadas em `runs_<fase>.csv` são puladas.
`--config configs/experiments/sa_calibration/smoke.yaml` roda o pipeline inteiro em ~1
minuto (instâncias de 12/16 setores, 3 mil iterações) — só para checar que funciona.

Para limpar: `rm -rf experiments/sa_calibration/outputs`.

## Result

Relatório completo, com tabelas, gráficos, limitações e próximos passos:
[`REPORT.md`](REPORT.md). Foram 1.534 execuções do SA; o MIP fechou (gap residual ≤ 1%)
nas 4 instâncias, em 1 a 5 s.

**Recomendação: `lhs_32`**
- Parâmetros: T0 ≈ 257, T_min ≈ 0,011, α derivado do orçamento (≈ 0,99995), β ≈ 0,77,
  γ ≈ 0,25, ρ ≈ 146.
- Revalidação (seeds 101–108; `base40`, `tight40` e `tight80`, esta nunca vista):
  **gap médio de 1,96%** (P90 4,6%, 24/24 viáveis).
- Para comparar: 23,8% do `default_repo` e 8,1% do default com α ajustado ao orçamento.

**Hipóteses**

1. **Confirmada.** Derivar α do orçamento é o maior ganho isolado (23,8% → 8,1%).
2. **Não confirmada.** T_min entre 0,001 e 1 dá resultados equivalentes. Com T_min ≥ 5 e
   ρ baixo, 1 em 10 execuções termina inviável. O que importa é o tempo em T ≈ 5–40:
   depois da iteração ~76 mil, nenhuma execução da recomendada melhorou.
3. **Não confirmada.** β e γ mexem ≈ 1 p.p. ou menos também em `tight40`, dentro do ruído.
4. **Confirmada, com ressalva.** ρ alto (≥ 2.000) proíbe atravessar regiões inviáveis e
   custa ~7 p.p. Mas o ganho está numa janela estreita: ρ < ~100 termina inviável e
   ρ = 500 já perde ~5 p.p. Esse valor precisa ser revalidado em escala maior antes de
   virar default.

O TPE não superou o melhor ponto do LHS: o objetivo é ruidoso demais (desvio de 2–3
p.p. com 5 seeds). Nesses tamanhos, o HiGHS resolve o MIP mais rápido que uma execução do
SA; o próximo passo é medir onde isso deixa de valer (200–800 setores).
