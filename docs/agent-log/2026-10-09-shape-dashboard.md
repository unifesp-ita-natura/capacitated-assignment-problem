# Standalone shape dashboard

**Date:** 2026-10-09

## Task

Build a front to explore shape backtest result folders while keeping level
entirely out of scope, including any optional integration.

## Outcome

A local shape-only dashboard selects a run, model, cycle, origin and sector.
It compares daily actual/predicted demand, cumulative shares and signed errors,
reports filtered and original metrics, and downloads configuration and points.

## What changed

Added `shape_dashboard/` with a read-only Python server, cached/filtering data
adapter, offline HTML/CSS/JS charts and a Portuguese usage guide. Added focused
adapter/API tests. No level or existing dashboard files were changed.

## Notes

Charts align sectors by relative day since each sector's window opening (day 1),
including when their calendar opening dates differ; chart shares are weighted by
actual volume. Filtered metrics retain the backtest's per-curve definitions.
Cycle and origin are explicit to avoid mixing multiple predictions for a cycle.
No new forecasting models are run from the front. The actual total scales
shape predictions only for evaluation. Empty runs and undefined shares have
explicit UI states.

Validation: 51 dashboard and shape-backtest tests passed, including HTTP assets,
API errors and CSV/YAML downloads. Data adapter coverage is 95%; server coverage
is 80%. Ruff and JavaScript syntax checks passed; all added Python functions
have Radon grade A. Browser automation could not start because the Windows
sandbox helper failed, so visual/interactive browser QA remains unverified.
The temporary preview used clearly named test fixtures, not experiment results.

Follow-up: daily, cumulative and error chart axes use relative day instead of
calendar date. Point tables and downloads retain calendar dates alongside
`relative_day`. Archived `window_start` determines the reference; older archives
without it fall back to the first recorded day per curve. Tests cover different
sector openings and a declared opening earlier than the first recorded row.
