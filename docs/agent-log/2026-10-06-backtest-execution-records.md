# Backtest execution archives

**Date:** 2026-10-06

## Task

Let users name the output folder and preserve results, configuration, model
parameters and feature lists for each backtest.

## Outcome

`paths.run_dir` selects a fresh archive folder with comparison and prediction
CSVs, the original YAML and a JSON manifest. Existing run folders are rejected
to prevent overwriting; legacy CSV destinations remain supported.

## What changed

Added metadata and archive helpers in `experiments/compare_forecasters/records.py`,
integrated run status handling into `run.py`, and added a complete sample YAML,
Portuguese guide and tests for defaults, hashes, failure handling and legacy outputs.

## Notes

The manifest contains configured parameters with defaults, not per-fold fitted
state. Git changes are identified but not archived. Legacy configs create a
unique archive alongside their usual CSV outputs. New named-folder configs
can omit the legacy paths. No real-data backtest is automatically executed.

Validation: 27 archive and comparison tests passed, including a real backtest
on the small fixture. Statement coverage: records 100%, run 91.2%. Ruff lint
and formatting passed, and all inspected functions have Radon grade A.
