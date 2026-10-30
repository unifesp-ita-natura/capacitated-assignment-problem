# Hierarchy feature block ablation

**Date:** 2026-10-27

## Task

Investigate whether existing predictors worsen management-city shared
distribution and reconciled forecasts, using controlled backtesting rather
than feature-importance rankings alone.

## Outcome

A dedicated executor replays the full six-block model and removes one block
at a time on shared past-only fold tables. It calls the existing learned-model
implementation and retains the active-city evaluation and historical fallbacks.

## What changed

`experiments/hierarchical_level/ablation.py` archives each variant's inputs,
features, fitted metadata, predictions and metrics and checks the full control
against the previous reference. A YAML fixes the same 20-cycle histories and
nine targets. Documentation explains interpretation and limitations. Tests
compare the cached learned path against the standard forecasting API, reject
future-label influence and verify archival and reference reproduction.

## Notes

Production model and feature construction algorithms are unchanged. All
selected LightGBM levels lose the same block together; ETS and historical
regularization remain. Individual variables and unseen holdout performance
are not inferred from block-removal results. Baselines are reused from the
archived reference because their predictions do not use ML feature blocks.

An additional 10-cycle configuration was prepared, but its execution was
rejected at the permission prompt and was not started or retried. This
analysis therefore evaluates feature blocks with 20 training cycles only;
it cannot establish which blocks explain the shorter-history degradation.

The 20-cycle run completed. The full control matched the archived forecasts
point by point. All variants retained 38,945 active points and 726 uncovered
points. Removing history predictors reduced shared city MAE 230.7443 →
230.5000 (0.106%, 5/9 cycles). Removing recent city shares improved 7/9 cycles
but only 0.078% overall. Removing intermittency reduced reconciled city MAE
235.0135 → 234.6292 (0.164%, 5/9). No large city gain was found.

The stronger finding is higher in the reconciliation hierarchy: removing
history predictors reduced management MAE by 6.03% (7/9 cycles) and regional
MAE by 8.67% (6/9), with essentially unchanged active-city MAE. Calendar
removal worsened active-city reconciliation despite improving the all-points
diagnostic. Findings are exploratory and level/objective dependent.

Full results: `docs/reports/management-city-feature-ablation.md`. Twenty-five
tests passed, new-runner coverage 93%, Ruff passed and all checked functions
remained Radon A. Default model settings were not changed.
