# Management-city hierarchy experiment

**Date:** 2026-10-27

## Task

Evaluate shared-participation distribution and reconciled volume forecasts
for cities aggregated within sales managements, with moving-average controls.

## Outcome

`city_grouping: management` supports region → management → city while retaining
the sector-city default. Management-city identities include region, management,
state and municipality, combining sectors before training and scoring.

## What changed

`hierarchy_config.py` defines grouping-aware keys and validates the parent;
`hierarchy_data.py` applies them to sums and scenario geometry. Evaluation
decodes matching identities and retains the positive-demand city policy.
Run metadata records city keys. A dedicated YAML and guide describe execution.
Tests merge two sectors, preserve homonyms and totals, verify exported keys,
retain unknown buyers, and check invariance to future demand changes.

## Notes

The experiment keeps the nine previous target origins, 20 closed cycles,
features and hyperparameters. Both hierarchy depth and leaf granularity change,
so raw MAEs cannot be compared directly to sector-city scores. Results are
archived separately under `2026-10-27_management_city_v01`.

The full run completed: 38,945 active city-cycle points, 33,930 quiet points
excluded. Distribution MA10 has MAE 233.22, WMAPE 49.9738%, Bias −12.9157%;
shared distribution has 230.74 / 49.7910% / −17.8930%; reconciliation has
235.01 / 50.3531% / −16.3590%. Twenty-three tests passed, combined relevant
coverage 96%, Ruff passed and all checked functions remained Radon A.

For a matched architecture comparison, old sector-city forecasts were summed
to management-city before positive-demand filtering. Data hashes, folds,
population and actual demand matched. New shared distribution slightly improved
MAE but worsened WMAPE/Bias; new reconciliation worsened all three. This is not
evidence that coarser training generally improves accuracy. Full results and
verification details are in `docs/reports/management-city-hierarchy.md`.
