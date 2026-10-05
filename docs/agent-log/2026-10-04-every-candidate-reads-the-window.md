# Every forecasting candidate answers (sector, cycle, opening date)

**Date:** 2026-10-04

## Task

Tasso asked that every model be keyed by sector, cycle **and the cycle's
opening date**, not by sector and cycle alone, so the scenarios exist for
all of them: "the forecast for the case where the cycle starts on day X".
Until now only the pooled ETS (with `cycle_days_exponent`) and LightGBM
(with `window_features`) read the window. Every other candidate returned
one value per (sector, cycle), and the harness copied it onto every
scenario.

Tasso chose to have the window always on, with no way to turn it off, even
though this changes the numbers older experiments would give on a rerun.

A first pass made every model scale by the window's length only. Tasso
rejected it: the point is **the same cycle starting on different dates**.
The scenarios are a range of days from the cycle's opening, and the
window's length stays fixed. With a fixed length, that first pass gave
every scenario the same number.

## Outcome

`model.opening_scenarios(cycles, days)` builds one scenario per (sector,
cycle) and day. Every candidate (naive, ARIMA, both ETS variants,
cycle_factor, LightGBM) answers each scenario with its own forecast, which
depends on the day the window opens and on its length.

Backtest on `v2_baseline`, MAE per cycle:
- cycle_factor improves by 1.4%, from the length.
- naive:last_value improves by 0.7%.
- naive:mean is unchanged (1,903 → 1,908).
- The opening-day factor moves every model by 0.1% or less.

On real data, the scenarios for one cycle vary by about ±5% across days in
the ETS and by ±0.5% in LightGBM.

## What changed

- `src/forecasting/model.py`:
  - `opening_scenarios(cycles, days)` builds the queries for the same cycle
    opening on each day in `days`, with the window's length fixed.
  - `opening_factors(history)` estimates a multiplicative effect per day
    after the cycle's opening. It uses log items with the sector and the
    cycle taken out (two-way demeaning), averaged by day, shrunk toward 0
    with a prior of `OPENING_DAY_PRIOR = 100` cycles.
  - `opening_adjusted(candidate)` is an adapter. It divides the history by
    each cycle's day factor, lets the wrapped candidate forecast, and
    multiplies each scenario by its own day's factor.
  - `cycle_days_exponent(history)` estimates β in items ~ cycle_days ** β.
    It is the pooled within-sector slope of log items on log length,
    clipped to [0, 1].
  - `window_scaled(candidate)` is a new adapter. It divides the history by
    cycle_days ** β, lets the wrapped candidate forecast that level, and
    multiplies each scenario by its own length to the β.
  - `per_sector` applies both adapters, so naive, ARIMA and the
    per-sector ETS get them.
  - `align_predictions` no longer copies a (sector, cycle) value onto every
    scenario. It raises if a candidate leaves out `window_start` or
    `cycle_days`.
- `candidates/cycle_factor.py` is wrapped in both adapters. The pooled ETS
  is wrapped in `opening_adjusted`, since it handles the length itself.
- `candidates/ets.py` and `src/config/schema.py`: the pooled ETS always
  scales by the window. `cycle_days_exponent` defaults to `pooled`, and a
  fixed value still needs a shared alpha.
- `candidates/lightgbm.py`, `features.py` and `schema.py`: the
  `window_features` flag is gone, and the three window features are always
  in the model.
- `configs/experiments/window_aware/all.yaml`: the window-blind rows are
  dropped, since they would now duplicate the window-aware ones.
  `experiments/window_aware/plot_level.py` asks about each sector's window
  from the same cycle one year earlier, shifted 52 weeks.
- Tests:
  - Fixtures carry `window_start` and `cycle_days`.
  - `test_scenarios.py` checks, for every registered candidate, that the
    same cycle opened on day 15 forecasts more than on day 0 when the
    history sold more on day 15. It also checks that a 14-day window gets
    less than a 21-day one.
  - With the day factor switched off, the day test fails for the six
    candidates that go through `opening_adjusted`.
- Notes in `experiments/README.md`, `v2_baseline` and `window_aware` say
  that their saved numbers come from window-blind models.

## Notes

- **The data barely measures the opening day.** Within a sector, the
  opening day varies by ±1.7 days on base_tratada_v2. Sectors stay in
  their sub-block. Once sector and cycle are taken out, the correlation of
  items with the opening day is −0.03.
  - The factors come out between 0.95 and 1.06 and jump from one day to the
    next (day 16: 1.056, day 23: 0.969, day 24: 1.052). That looks like
    noise, not a calendar effect.
  - Days 2 to 6 never occur, so their factor is 1.
  - A smoothed curve over the days, or a "distance from the sector's
    usual day" term, are the next things to try if the optimizer needs a
    real signal here.
- **LightGBM is left as it was.** Its window features (gap since the
  previous opening, day of the month) already change with the opening day.
  The day after the cycle's opening stays out of its features, because
  with the sector it encodes the sub-block. Across 30 days its answer moves
  by ±0.5%.

- **Why a length law as well.** A per-sector series
  model has no input through which an opening date could enter. What the
  date changes for the sector is how long its window lasts. The
  `window_aware` experiment found that the gap since the previous opening
  and the day of the month move nothing within ±2 days. One law, applied
  in one adapter, covers all those models without touching their code.
- **Measured β on base_tratada_v2:**
  - The within-sector estimate gives 0.32 on the full panel and 0.18 to
    0.33 across the 11 folds.
  - The pooled ETS, which picks β by forecast error, chose 0.1 to 0.2.
  - Adding cycle effects (two-way) gives 1.13. Within a cycle, length
    varies only with the sub-block, so that estimate measures "late
    sub-blocks sell less", not the effect of length. It was rejected.
- **Fixed β sweep, MAE per cycle on v2_baseline's split (6,736 points):**

  | candidate | β = 0 (before) | 0.1 | 0.2 | 0.3 | estimated |
  |---|---|---|---|---|---|
  | cycle_factor:last1 | 1,836 | 1,822 | 1,813 | 1,808 | **1,810** |
  | naive:last_value | 2,432 | 2,421 | 2,414 | 2,411 | **2,416** |
  | naive:mean | 1,903 | 1,902 | 1,903 | 1,907 | **1,909** |

  Per CD-day the picture is the same: cycle_factor goes from 5,399 to 5,359,
  and naive:mean from 5,824 to 5,840. The models that track the recent level
  gain. naive:mean averages over many windows, so it barely needs the scale.
- **Side finding, not pursued:** with the window, `cycle_factor:last1`
  (1,810 per cycle, 5,359 per CD-day) is level with LightGBM (1,802 /
  5,384) on v2 and beats it per CD-day. The `cycle_factor` experiment had
  rejected it on the one-year base. It is worth a run of its own.
- **Not touched:** `level.py`, `scoring.py` and `forecast_future_cycles` in
  `model.py`. They are the older pipeline on the synthetic generator, not
  the candidates the harness ranks.
- **Open:** the backtest only scores the window that actually happened. No
  run can check whether the answer for an opening date that never happened
  is right.
