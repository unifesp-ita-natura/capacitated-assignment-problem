# Combined city-day demand forecast

**Date:** 2026-10-16

## Task

Connect predicted city-cycle level with sector daily shape for explicit opening
scenarios and provide daily historical validation with fixed configurations.

## Outcome

An independent executor produces level, shape and combined daily predictions,
preserving sector, state, city, cycle and opening scenario. Historical replay
uses the same production function, then measures actual city-day demand errors.

## What changed

Added combined input, forecasting and validation modules; a YAML-driven runner;
future and validation example configurations; integration tests and a guide.
Existing level models and backtests were not edited. The shape history builder
accepts an optional explicit calendar to avoid recomputing completeness after
restricting the training sample.

## Notes

Production uses only closed, complete cycles available before the forecast
timestamp. Scenario paths are isolated, with explicit intermediate windows
when necessary for recursive forecasts. Missing levels remain missing; shares
and city totals are checked for conservation. Replay reports uncovered demand,
including new cities. Recorded opening schedules are assumed available at the
historical cutoff: no versioned schedule exists to verify that assumption.
Future dates are deliberately not invented; the future schedule template
requires the user's planning inputs.

Validation: the initial integration/regression run passed 88 tests; the final
dedicated suite passed 33 integration tests, including native LightGBM shape
and ETS, ARIMA and LightGBM level candidates. New forecasting/input/replay logic
has 96%/94%/98% test coverage. Ruff passed; every new function has Radon grade A.

A fixed naive-mean + sector-curve replay of real cycle 202612 was saved under
`experiments/forecast_combined/outputs/validation_v01`: 243,183 scored city-day
points, 98.9311% actual-volume coverage, daily MAE 20.006725 items and WMAPE
1.796623. These are baseline diagnostics, not model selection results.
The first real run exposed empty-city metric warnings; the final code leaves
those metrics undefined explicitly, verified by a regression test. The provided
validation YAML now points to a fresh `validation_v02` directory so users can
execute it without overwriting the saved check.
