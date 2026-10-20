# Level backtest diagnostics

**Date:** 2026-10-06

## Task

Report Bias, WMAPE and existing diagnostic metrics alongside the level MAE ranking.

## Outcome

The comparison CSV reports RMSE, MASE, P90 absolute error, worst-fold MAE,
Bias in items, Bias percentage and WMAPE on the common prediction subset.
Level ranking remains MAE; model fitting and temporal splits are unchanged.

## What changed

Added bias and relative-error helpers in `metrics.py`, common-subset diagnostics
in `comparison.py`, focused numerical tests, and a Portuguese metric guide.

## Notes

MASE retains each fold's training-only scale. Percentage bias uses percentage
units; WMAPE uses a ratio. Zero actual total yields NaN for relative metrics.
Daily ranking retains its existing behavior. Validation: 45 focused metrics,
comparison and evaluation tests passed; metrics and comparison modules have
100% statement coverage. Ruff formatting/lint passed; all inspected functions,
including tests, have Radon grade A. Checks used `uv run --no-project` with the
existing `.venv` to avoid installing solver dependencies. No real-data backtest
was executed as part of this report change.
