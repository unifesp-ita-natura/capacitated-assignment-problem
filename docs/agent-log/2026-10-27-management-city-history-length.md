# Management-city training history length

**Date:** 2026-10-27

## Task

Compare management-city shared distribution and reconciliation with 10 versus
20 closed training cycles, reporting 10- and 20-cycle moving averages separately.

## Outcome

A separate 10-cycle YAML fixes the same nine target cycles and all model
parameters. The completed 20-cycle reference is reused rather than retrained.
No forecasting algorithm or existing configuration is modified.

## What changed

`management_city_history10.yaml` selects 10 training cycles, the same six
feature blocks and baselines of 3, 5 and 10 cycles. The 20-cycle moving-average
reference comes from the existing 20-cycle run, avoiding a baseline window
larger than its available training history.

## Notes

Evaluation retains positive-demand management-city-cycle observations and
counts unknown buyers as errors. Paired verification checks data hashes,
features, parameters, cutoffs, history suffixes and actual city populations.
Direct moving averages and historically distributed variants are reported
separately: historical share smoothing can use all available training cycles,
even when the moving-average window is shorter.

The run completed on the same 38,945 active points, 726 uncovered points,
data hashes, target cutoffs, model settings and feature list. Metrics were
independently recomputed. Shared distribution MAE rose from 230.74 (20)
to 236.33 (10); reconciliation rose from 235.01 to 242.36. Both lost to
distributed MA20 (232.93) with the shorter training history. Shared
distribution with twenty retains the lowest MAE, while MA20 has lower Bias.

Comparison CSVs and checks are archived under
`2026-10-27_management_city_history_comparison_v01`. Full discussion:
`docs/reports/management-city-history-length.md`. No forecasting source code
was modified; the reference was reused and no unit tests were repeated.
