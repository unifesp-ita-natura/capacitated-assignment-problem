# Region and sales-management level features

**Date:** 2026-10-06

## Task

Use the demand base's `CD_RE` region and `CD_GV` sales-management codes as
additional level predictors.

## Outcome

LightGBM supports optional categorical organization features. A separate YAML
compares baseline, region, management and both without changing the user's
existing experiment configuration.

## What changed

The loader preserves identifiers as text and the panel validates sector-cycle
assignments. Feature selection is configurable; model frames share historical
category vocabularies. Recursive prediction carries the latest non-null
historical sector assignment. Added focused tests and a Portuguese guide.

## Notes

The inspected real base has 8 regions, 37 management codes and no within-sector
changes or missing codes. Historical assignments may change between cycles;
forecasting uses the latest observed assignment rather than holdout metadata.
Planned future reassignment input is outside this change. No solver is required.

Validation: 104 targeted tests passed, Ruff format/lint passed and all inspected
functions have Radon grade A. Statement coverage: features 100%, dataset 91.2%,
LightGBM candidate 94.4%. The comparison YAML validates and the real panel loads
18,978 rows with both organization codes. The complete backtest was left for
the user to execute.
