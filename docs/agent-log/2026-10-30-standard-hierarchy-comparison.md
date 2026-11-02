# Simplify the default hierarchical backtest comparison

**Date:** 2026-10-30

## Task

Keep only a pure five-cycle moving average, learned distribution and learned
reconciliation in the standard hierarchy comparison. Preserve existing model
results, historical reproducibility, combined forecasting and solver code.

## Outcome

`HierarchySettings` defaults to `comparison: standard` and
`baseline_windows: [5]`. Standard runs emit exactly `direct_ma_5`,
`distribution_ets_lgbm` and `reconciliation_ets_lgbm`; the pure baseline
does not execute distribution or WLS. Twenty-cycle training is unchanged
in the supplied standard YAMLs.

## What changed

The previous candidate set remains available with `comparison: historical`.
Original full, management-city and recent-share configurations were copied
to `configs/experiments/hierarchical_level/historical/` before updating
the standard YAMLs to new output paths. Existing ablation/history experiments
explicitly request historical behavior. Old ablation manifests without the
new field remain compatible as historical references, with all other
settings still checked strictly.

Regression tests compare every retained forecast and metric against the
historical path for both city groupings. A baseline test rejects any attempt
to distribute or reconcile the pure mean. Documentation distinguishes
standard use from historical experiments. Existing output artifacts were
not rewritten; no optimization, SA, level/shape integration or feature
engineering functions were changed.

## Notes

The direct learned-volume estimates are still required internally to
reconcile forecasts. Only their separate diagnostic output is omitted
from the standard comparison. Historical mode retains that diagnostic.
Validation: 75 tests passed across hierarchy, combined level/shape forecasting
and the shared harness. Retained standard predictions and metrics match the
historical path point by point for sector-city and management-city groupings.
Coverage is 94.6% for hierarchy orchestration, 97.1% for hierarchy settings
and 94.2% for the ablation runner. Ruff format/lint passed, changed Python
functions remain Radon A, and `git diff --check` passed. Optimization, SA,
pipeline and combined forecast sources still match origin/main.
