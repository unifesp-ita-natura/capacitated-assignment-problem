# Re-score As-Is and To-Be on the realized demand

**Date:** 2026-11-02

## Task

The pipeline holds out 6 cycles, so their real orders exist. Evaluate the
forecast-optimized assignment and the As-Is calendar on those actuals, to
connect the forecasting and optimization halves into one end-to-end claim.

## Outcome

`python -m src.main --solver sa` (568 sectors, 6 held-out cycles, 20% churn)
now also reports the realized comparison. Balance metric on the forecast:
As-Is 47,583 -> To-Be 37,077 (22.1%). On the realized demand:
As-Is 132,522 -> To-Be 107,338 items, a 19.0% improvement. The gain mostly
survives forecast error (WMAPE 1.73 at city-day level, bias -5.4%); no
capacity violations either way. The absolute metric is ~2.8x larger on actuals
because real days are noisier than the smoothed forecast.

## What changed

- `src/pipeline/realized.py`: builds actual (sector, cycle, relative day)
  curves from the raw base and swaps them into the forecast instance
  (same sectors, calendar, capacities, As-Is); `rescore` evaluates both
  assignments with the same evaluator the solvers use.
- `explain.py`: `Explanation` gains `realized_as_is/realized_to_be`,
  rendered in the report and written to `pipeline_explanation.csv`.
- `main.py` wires the stage in; tests added in `tests/pipeline`.

## Notes

- Single SA run, seed 42; not yet repeated over seeds or for `mip`.
- Realized demand outside the instance's days is dropped.
- Realized `relative_day` uses the sector's earliest `Dt Abertura` per cycle,
  matching `recorded_windows`.
