# Daily shape forecasting and independent backtest

**Date:** 2026-10-09

## Task

Forecast shape per sector-cycle-day rather than sector-cycle-CD-day and provide
an independent backtest without changing level code.

## Outcome

A separate YAML-driven experiment evaluates uniform and historical sector
curves, aggregates CDs and includes zero-sale days. Actual cycle totals scale
the daily shares only during scoring, isolating shape error from level error.

## What changed

Added `shape_daily.py`, `shape_evaluation.py`, `experiments/compare_shapes`,
an example YAML, focused tests and a Portuguese execution guide stored as
`docs/guides/daily-shape-backtest.md`, following English filename conventions. Each run
archives predictions, metrics, its YAML and model/input metadata. Existing
level and legacy daily modules were left intact.

## Notes

Ranking uses share MAE in percentage points with equal weight per curve.
The split mirrors the cycle-based expanding/sliding workflow; actual historical
windows are evaluated, without alternate opening scenarios. Unknown sectors
or projections with no usable mass fall back to uniform. Zero-total curves
have undefined actual shares and are separately counted. Aggregate bias is
zero by conservation and cannot discriminate shapes.

Validation: 60 focused shape and existing daily/level tests passed; Ruff passed,
all new functions have Radon grade A. New shape logic/evaluation coverage is
92%/98%. An in-memory real-base backtest evaluated 11 origins, 6,737 curves and
132,991 daily points per candidate; share sums differ from 1 by at most
2.22e-16. Sector/uniform share MAE was 6.633565/6.640746 percentage points.
This validation did not create a user experiment output folder.
