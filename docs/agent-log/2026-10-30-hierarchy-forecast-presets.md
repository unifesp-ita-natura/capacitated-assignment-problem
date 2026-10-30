# Hierarchical presets for future daily forecasts

**Date:** 2026-10-30

## Task

Make the selected sector-city level model the forecast default and allow the
selected management-city model without breaking the existing daily shape output.

## Outcome

The combined runner and application configuration default to ETS regional totals
and learned hierarchical shares. A grouping switch chooses sector-city with all
feature blocks or management-city without the history block.

## What changed

Added a forecasting-only distribution entry point using the same formulas as the
hierarchy backtest, closed-cycle recursive projection and a conservative historical
allocation from management-city back to sector-city. The runner records effective
feature blocks and hierarchy settings. Updated example configurations and the
combined forecasting guide; optimization and SA code were not changed.

## Notes

Presets reference archived positive-demand city metrics, not a new model search.
Management-city metrics do not measure the new allocation to sectors. Training
uses twenty closed cycles by default; historical shares use five cycles. Tests
cover backtest equivalence, daily conservation, scenario isolation, recursive
projection, historical fallback and exclusion of future demand.
Validation: 72 existing forecasting tests and 16 new integration tests passed;
Ruff checks passed and changed Python functions remain complexity grade A.
