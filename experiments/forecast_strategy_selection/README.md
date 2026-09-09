# forecast_strategy_selection

**Status:** exploratory

**Goal:** Which level (campanha-total) and shape (within-window curve)
strategy combination forecasts a held-out campanha's volume and daily order
curve most accurately?

**Hypothesis:** `linear_trend` beats `naive_last` for the level forecast
because the synthetic generator applies a per-campanha demand factor with a
mild trend; `recency_weighted` (swept over a few half-lives) beats
`plain_average` for the shape forecast because sector window curves are
static in the generator, so any strategy should score similarly and the
comparison mainly exercises the harness rather than separating strategies.

**Design note:** Uses `src.generate.generator` to produce 6 campanhas of
synthetic orders for a small sector set (as-is block/subloco assignment,
fixed seed). The last campanha is held out as ground truth; every earlier
campanha is history. Level strategies are scored via
`src.forecasting.scoring.score_level_strategies` against the held-out
campanha total; shape strategies (including the `recency_weighted` and
`shrinkage` sweeps from `src.forecasting.sweep`) are scored via
`score_shape_strategies` against the held-out daily order curve. Known
limitations: synthetic data only (no real `data/raw/Unifesp_Demanda.csv`
wiring yet), single seed, small sector count for runtime, and
`level_holt_ets`/`level_arima`/`level_calendar_regression` are skipped
automatically when the optional `statsmodels` extra isn't installed
(`uv sync --extra forecast`).

```bash
uv run python -m experiments.forecast_strategy_selection.run
```

Writes `experiments/forecast_strategy_selection/outputs/level_scoreboard.csv`
and `.../shape_scoreboard.csv` (gitignored) plus prints the best combination
and its overall daily WMAPE.
