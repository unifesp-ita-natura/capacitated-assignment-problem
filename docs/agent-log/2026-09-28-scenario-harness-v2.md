# Scenario queries in the forecasting harness, on base_tratada_v2

**Date:** 2026-09-28

## Task

The team needs a function f(cycle length in days, opening day, sector) that
returns the expected items for the cycle. The optimizer will call it for
every opening date it weighs, combine the answer with Ana's shape curve,
and choose each sector's opening date. Turning (sector, d, n) into a model's
own inputs is the model author's job, not the harness's. This work covers
the harness side and the move to `data/base_tratada_v2.csv`.

## Outcome

- The harness answers scenario queries through `model.forecast(candidate,
  history, queries)`.
- Every existing candidate works unchanged. For each of them f is flat
  across opening days.
- A rerun of `ets_level/shared_alpha_one_year` reproduces the saved numbers
  exactly.
- On v2, LightGBM leads the pooled ETS by 1.1% per cycle and 0.4% per
  CD-day. Their average beats both per cycle by 2.3%. See
  `experiments/v2_baseline/`.

## What changed

- `src/forecasting/dataset.py`:
  - The loader rebuilds `Qtde dias` when the base lacks it.
  - It drops orders that v2 repeats under several opening dates, keeping
    the copy with the latest opening date.
  - `build_item_panel` now carries each sector's own window (`window_start`,
    `cycle_days`). `_sector_windows` is shared with `build_daily_base`.
- `src/forecasting/model.py`:
  - `QUERY_KEYS` and `align_predictions`: predictions keyed only by
    (sector, cycle) apply to every scenario of that cycle.
  - `forecast`, the scenario entry point.
  - `per_sector` no longer treats scenarios of one cycle as extra forecast
    steps.
- `src/forecasting/evaluation.py`: each target is queried with the window
  the sector actually had.
- `tests/forecasting/test_scenarios.py`: new tests.
- `experiments/v2_baseline/` and `configs/experiments/v2_baseline/all.yaml`.

## Notes

- **The fan-out in v2.** It affects 11 sector-cycles, 1,190 rows and 163,550
  items. The copies differ only in `Dt Abertura` and the columns derived from
  it. The rebuilt `Qtde dias` differs too, so it has to be excluded from the
  dedup key. A first version missed this, and the test caught it.
- **The fan-out looks like a join against every opening date the sector had
  in the cycle,** filtered to openings on or before the order date. Keeping
  the latest opening leaves each sector's window starting at its first
  order. This still needs confirmation from whoever built v2.
- **Next step: a window-aware candidate.** Start with the cycle length:
  forecast items per window day, then multiply by the queried `cycle_days`.
  After that, try calendar variables for the opening date. Score both
  against `v2_baseline`.

## Follow-up: window-aware ETS and LightGBM (same day)

The team chose the models: the scenario forecast is the ETS and LightGBM
already in the harness, made to read the window.

- **The data changed the plan.** 14-day windows sell 0.97 of a sector's mean,
  not the 0.67 that "items per day × days" implies. So the ETS got an
  exponent, `items ∝ cycle_days ** β`, with β estimated alongside α instead of
  fixed at 1. The pooled β came out at 0.1 to 0.2.
- **LightGBM got three window features:** the length, the days since the
  sector's previous opening, and the day of the month. The offset from the
  cycle's opening was left out, because it encodes the sub-block.
- **Result (`experiments/window_aware/`):** LightGBM improved by 0.7% per cycle
  and the ETS by 0.3%. All of the gain is on short windows, where they
  improved by 3.5% and 2.1%. Across ±2 opening days, neither model's answer
  moves; only the length does.
- **`model.forecast` now requires `CICLOS` and `opening_date` in each query.**
  The earlier default label `"next"` broke LightGBM, which reads the cycle
  number from the code, and it would have misled any model that does the
  same.
- **The first LightGBM window test** alternated lengths in a fixed pattern.
  The lags learned that pattern, so the model never needed `cycle_days` and
  both scenarios came out equal. Random lengths fixed the test.
