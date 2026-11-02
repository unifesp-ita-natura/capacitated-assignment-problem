# Recent city participation features

**Date:** 2026-10-24

## Task

Add city-to-sector participation over the last one, two and three closed
observations and compare an otherwise identical hierarchy backtest.

## Outcome

The optional `city_recent_shares` block provides `share_1`, `share_2` and
`share_3` only at city level. Existing default feature selections and archived
reference results remain unchanged.
The full nine-origin comparison completed: city distribution WMAPE changed
from 72.2353% to 72.2421%; reconciled city WMAPE changed from 72.2365% to
72.2139%, with worse absolute Bias. No material city improvement was observed.

## What changed

`hierarchy_features.py` calculates ratios of sums and applies the existing
closed-cycle feature cutoff. `hierarchy_config.py` accepts the new block.
`recent_city_shares.yaml` enables it with the same full-backtest parameters.
The guide explains the optional block and reproduction command; tests cover
ratio windows, overlapping cycles, zero denominators and quiet cities.

## Notes

Evaluation includes zero-demand cities as before. Filtering positive demand
is a supplementary diagnostic, not a change to training or prediction.
Reference run: `2026-10-24_full_v02`; experiment:
`2026-10-24_recent_city_shares_v01`. Full report:
`docs/reports/hierarchical-city-recent-shares.md`. Paired CSVs and equality
checks are archived in the experiment folder. All baseline results, input
hashes, folds, parameters except the feature selection, and scoring coverage
matched. Eighteen hierarchy tests passed; targeted feature coverage was 92%.
Ruff passed and Radon complexity remained grade A. The optional block remains
disabled in existing configurations and defaults.
