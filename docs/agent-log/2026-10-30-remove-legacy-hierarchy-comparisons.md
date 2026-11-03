# Remove obsolete hierarchy baseline execution

**Date:** 2026-10-30
**Related:** `69b9ece` (safety checkpoint before removing any legacy code)

## Task

Commit the working forecasting implementation first, then remove distributed
and reconciled moving averages, multiple baseline windows, historical mode
and their dedicated execution configurations. Retain the pure five-cycle
mean, learned distribution/reconciliation, feature analysis and combined
level/shape forecasting. Do not edit optimization or simulated annealing.

## Outcome

Created checkpoint commit `69b9ece` after installing the repository hook and
passing its format, lint and Radon checks. The runtime now exposes exactly
three comparison candidates and no historical execution switch. Model
training remains 20 cycles in the provided configurations; five cycles is
the fixed reference window, now recorded in `features.json`.

## What changed

Deleted the legacy baseline builders and their settings/validators, and
removed five obsolete YAMLs. The current ablation runner still accepts
learned-model references from archived runs by ignoring only the retired
baseline-only settings; all training/model/data checks remain strict.
Updated the active guides without rewriting old reports or generated results.

## Notes

Before removal, saved pointwise predictions and metrics for sector-city
and management-city fixtures. After removal, all 81 predictions and 21
metric rows matched within 1e-9 absolute and 1e-12 relative tolerance.
The local regression files are ignored under `.venv/hierarchy_cleanup_regression/`.
The previous implementation and configurations remain recoverable from
the checkpoint commit. Validation: 72 tests passed across hierarchy,
combined level/shape forecasting and the shared harness. Ruff format/lint
passed; changed Python functions remain Radon A. `git diff --check` passed.
Git confirmed solver, calibration, scaling, pipeline and combined forecast
sources remain identical to origin/main. Archived reports were not edited.
Coverage is 92.7% for hierarchy orchestration, 100% for hierarchy settings
and 94.2% for the ablation runner.
