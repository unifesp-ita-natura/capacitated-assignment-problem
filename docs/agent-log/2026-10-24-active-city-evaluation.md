# Active-city hierarchy evaluation

**Date:** 2026-10-24

## Task

Make positive-demand sector-state-city-cycle observations the primary city
evaluation population, following the user's definition of activity.

## Outcome

Hierarchy metrics now exclude city observations with actual demand zero by
default. Full-point diagnostics remain available separately. Training,
rosters, forecasts, historical feature zeros and upper-level totals are unchanged.

## What changed

`summarize()` in `hierarchy_evaluation.py` filters only city metrics and
reports total, evaluated and excluded point counts and scoring scope. Empty
active groups have missing metrics, not zero error. Unknown positive-demand
cities remain scored. `run.py` archives full-point diagnostics and records
the activity definition. Predictions retain all observations and an explicit
`included_in_primary_metrics` flag.

## Notes

Activity here means observed positive demand, not independently verified
commercial activity known at forecast time. The criterion is used only for
evaluation; filtering forecast rosters by target purchases would leak future
information. Archived runs are rescored into separate `active_city_analysis`
folders, preserving original artifacts and without retraining. Tests cover
unknown buyers, inactive groups, unchanged parents and non-observed scenarios.

Both archived runs were successfully rescored: 43,951 active city-cycle
points retained and 56,318 zero-demand points excluded per candidate. In the
recent-share run, distribution ETS/LightGBM has MAE 220.3063, WMAPE 57.0424%
and Bias −21.7222%. The lowest-WMAPE moving-average baseline is now
reconciliation MA20 (57.3036%), rather than distribution MA3 under the
previous full-point criterion. The model forecasts themselves did not change.

Validation: 20 hierarchy tests passed with 97% evaluation-module coverage.
After simplifying two test assertion groups, both focused tests passed again;
Ruff passed and all checked functions have Radon grade A.
