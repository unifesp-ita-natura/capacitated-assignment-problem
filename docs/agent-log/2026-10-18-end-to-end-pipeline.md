# End-to-end pipeline in `src/main.py`

**Date:** 2026-10-18
**Related:** branch `luiza/mip-size`

## Task

Luiza asked for `main` to run raw data -> forecast model -> daily forecasts ->
chosen solver (MIP, matheuristic or SA) -> explanation of the metric. Before
this, `src/main.py` ran only on synthetic data, offered no matheuristic, ignored
its `config_file` argument and never explained the result.

## Outcome

`uv run python -m src.main [--config ...] [--solver mip|matheuristic|sa] [--max-sectors N]`
reads `data/base_tratada_v2.csv`, holds out the last `horizon_cycles` complete
cycles, forecasts them with the combined level x shape model using only data
before the cutoff, builds a `ProblemInstance`, solves it and prints/writes an
As-Is vs To-Be report. On a 30-sector slice MIP and the matheuristic reach the
same optimum (34.9% better than As-Is). On all 607 sectors SA improves the
metric 4.7% in 0.3 s (the full run takes ~27 s, nearly all forecasting).

## What changed

- `src/pipeline/` (new): `forecast_stage.py` (horizon plan, forecast, replayed
  WMAPE), `solve_stage.py` (`SOLVERS` dispatch returning result + assignment),
  `explain.py` (As-Is vs To-Be metric, utilization, churn, forecast accuracy).
- `src/solver/real_inputs.py` (new): the real-data bridge to `ProblemInstance`.
- `src/config/schema.py`: `PipelineConfig` under `RunConfig.pipeline`;
  `configs/pipeline.yaml` is the example.
- `block_assignment.solve_with_assignment` returns the MIP assignment;
  `combined_validation.recorded_windows` made public for reuse.
- Outputs in `results/`: `pipeline_solve.csv`, `pipeline_explanation.csv`,
  `pipeline_assignment.csv`.

## Assumptions to revisit

- The base has no block/sub-block columns, so As-Is slots are inferred from
  each sector's `Dt Abertura` relative to the 5th percentile opening of the last
  training cycle, clipped to 15 slots. This is an approximation.
- Sector -> CD is the CD with most items. The forecast is shifted by calendar
  days from the slot, as in the synthetic pipeline (weekends not modeled).
- The horizon is a backtest on recorded windows, so forecast accuracy is real;
  forecasting truly future cycles needs a window schedule from the client.
- Forecast WMAPE is city-day level and high (about 1.8 with naive:mean).

## Follow-up: balance window and overlapping cycles

The first version laid each cycle in its own `15 + L - 1` day block and measured
max - min over every day, so each block's ramp-up/ramp-down (not the assignment)
dominated the metric. Now `real_inputs` spaces cycles by the median real opening
gap (~21 days, so they overlap and `build_projected_demand` sums shared days) and
`inputs.balance_days_by_cycle` drops only the horizon's first and last 14 days from
the objective; capacity still covers every day. SA marks excluded days `-1`. On all
568 sectors, 6 cycles, SA: As-Is metric 176,561 -> 56,729 items (45.7% of mean
load), To-Be 43,209, 23.8% better than As-Is (was 6.9%). The synthetic
calibration instances keep the old non-overlapping layout.
