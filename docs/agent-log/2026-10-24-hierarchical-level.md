# Isolated hierarchical level experiments

**Date:** 2026-10-24

## Task

Compare hierarchical distribution and reconciliation with moving-average
baselines and hybrid ETS/LightGBM models. Preserve shape and existing level
executors. The user selected histories closed before the target first opening.

## Outcome

Added a configurable hierarchy from region through management and sector to
state-identified cities. Experiments archive per-level and per-cycle MAE,
WMAPE, Bias, coverage, planned scenarios, settings and fitted-model metadata.

## What changed

New `hierarchy*.py` modules own node aggregation, time-aware features, temporal
root totals, pooled internal models, distribution and sparse nonnegative WLS.
An isolated runner, YAML examples, tests and a Portuguese guide describe the
workflow. SciPy is declared explicitly in the forecast extra.

## Notes

Training features skip overlapping unclosed cycles too. New target cities
are scored with zero prediction and missing coverage; their demand is not
dropped. Alternative opening layouts have no invented actuals. WLS scales by
historical demand standard deviation; it is not MinT. Parent scenario geometry
uses equal sector weights. Historical dates are treated as known plan inputs;
operational use requires an independently known plan and roster.

Validation: 14 hierarchy tests passed with 96% statement coverage; the broader
forecasting suite passed 293 tests before the final two boundary tests were
added. Ruff passes and all new functions meet complexity grade A.

Real-data execution `experiments/hierarchical_level/outputs/2026-10-24_full_v02`
completed nine origins (202604–202612), 15 candidates and 1,592,745 prediction
rows. Independent checks confirm closed histories, conserved totals and metric
recalculation. Reconciliation ETS/ML wins WMAPE at region (10.73%) and management
(14.95%); distribution ETS/ML wins at sector (41.09%). City WMAPE is lowest for
distribution MA3 (71.46%), while reconciled ETS/ML reduces aggregate Bias to
−0.55%. These are exploratory comparisons, not a final independent holdout.

The first full run was interrupted for performance and marked failed. Node IDs
now encode each distinct identity once, and calendar/child aggregations are
vectorized. The failed attempt is preserved separately as `full_v01`.
