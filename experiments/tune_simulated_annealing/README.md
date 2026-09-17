# tune_simulated_annealing

**Status:** exploratory

**Goal:** Which combination of SA search-dynamics hyperparameters
(temperature schedule, stagnation stopping rule, sector/destination move
biases) minimizes the final energy the heuristic converges to on a
representative instance?

**Hypothesis:** The current `AnnealingParams` defaults in
`src/solver/heuristics/simulated_annealing.py` are placeholders ("a calibrar
na etapa de validação" per that module's own docstring), not calibrated
values. We expect Optuna's TPE sampler to find a temperature schedule
(`initial_temperature`, `cooling_rate`, `min_temperature`) that reaches a
lower final energy than the defaults within the same iteration budget,
since the defaults were picked without any search.

**Design note:** Uses `src.generate.generator` + the forecast pipeline
(`score_and_select_strategies`, `forecast_future_cycles`) + `src.solver.inputs`
to build one fixed synthetic instance (same defaults as `src/main.py`: 40
sectors, 21-day cycles, 6 historical cycles, 1 forecast horizon, seed 42),
built once and reused across all trials. Each trial suggests an
`AnnealingParams` (see `run.py:suggest_params` for the swept fields and
ranges) and is scored by the mean `AnnealingResult.energy` over 3 fixed
seeds (`SEEDS` in `run.py`) — the same quantity SA itself minimizes
(objective + capacity/churn penalties).

Held fixed rather than swept:
- `max_iterations` is capped at 20,000 for the sweep (vs. the 100,000
  production default) so ~50 trials × 3 seeds finishes in reasonable time.
  Top candidates should be re-validated at the full iteration budget before
  being adopted as new defaults.
- `penalty_coefficient` is left at its current default — it's a
  problem/objective-scaling knob, not a search-dynamics one.

Known limitations: single synthetic instance (results may not generalize to
very different sector counts or capacity profiles), only 3 seeds per trial
(objective is noisy), and TPE search uses a single fixed sampler seed rather
than repeated studies.

```bash
uv sync --extra tune
uv run python -m experiments.tune_simulated_annealing.run
```

Prints the best mean-energy value and its `AnnealingParams`, and writes
`experiments/tune_simulated_annealing/outputs/trials.csv` (gitignored) with
one row per trial (params, value, and the `objective`/`capacity_penalty`/
`churn_penalty` diagnostics recorded as user attributes).

## Result

Not yet run for real — fill in after a full sweep, including whether the
best-found params were promoted into `AnnealingParams`' defaults.
