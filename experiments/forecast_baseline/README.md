# forecast_baseline

**Status:** exploratory — see `configs/experiments/forecast_baseline/naive.yaml`
for the full goal/hypothesis/design-note/status block.

**Goal:** run the required baseline forecasting candidate (naive) end-to-end
through the shared evaluation harness (`src/forecasting/evaluation.py`)
against the real demand base, and report its rolling-origin MAE per sector.
This is the "at least one candidate producing a real error on a real sector"
milestone from the forecasting kickoff plan (see `docs/agent-log`).

**Hypothesis:** `last_value` won't be competitive on its own, but it
establishes the number every future candidate (SARIMA, LightGBM, ...) has to
beat, and its MASE should land near 1.0 since it's effectively the same
one-step benchmark MASE scales against.

```bash
uv run python -m experiments.forecast_baseline.run
```

Reads `data/base_tratada.csv` (not committed — see `data/README.md`),
aggregates it with `src/forecasting/dataset.build_item_panel` (items per
sector per cycle, dated by cycle opening date, cycles the base only
partially observed dropped), and runs `naive:last_value` through
`RollingOriginSplit(horizon=1, min_train_cycles=6, window="expanding")`.

Writes `outputs/naive_errors.csv` (gitignored) — one row per scored
(sector, cycle, origin) with the actual, the prediction, and the absolute
error — and prints the aggregate summary (MAE, RMSE, MASE, p90, worst fold).

## Result (2026-09-14)

| metric | value |
|---|---|
| MAE | 2,513 items |
| RMSE | 3,702 items |
| MASE | 1.06 |
| p90 abs error | 5,379 items |
| worst-fold MAE | 2,764 items |
| sectors scored | 623 / 633 |
| origins (folds) | 6 |

## Known limitations of this run

- **Only 6 rolling-origin folds.** The base has one year of history; after
  dropping the two cycles it only partially observed (202613, 202614 as of
  this export) and reserving 6 cycles for `min_train_cycles`, only cycles
  202607–202612 get scored. Treat the numbers above as a first read, not a
  settled baseline — they'll firm up as more cycles accumulate.
- **No seasonal signal available.** With a single year of history,
  `seasonal_naive` and any seasonal component in future candidates (SARIMA,
  ETS) can't be evaluated yet — see `docs/agent-log` for the fuller
  reasoning.
- **10 sectors never scored** (623 of 633): sectors with too little history
  at a given origin are skipped by the candidate, not scored with a
  fabricated value — see `src/forecasting/model.InsufficientHistoryError`.
