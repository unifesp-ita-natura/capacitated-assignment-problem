# Daily output for the forecasting harness

**Date:** 2026-09-26

## Task

The harness grouped everything by cycle: it forecast and scored one number
per (sector, cycle). Capacity is given in items per day and the MIP consumes
per-day demand, so the harness had to output and score a number per day.

## Outcome

`evaluate()` can now also spread each cycle forecast over the days of the
sector's window and split it across the sector's CDs, scoring it per
sector-day and per CD-day. Candidates are unchanged and the per-cycle numbers
are identical. Ranked per CD-day, `naive:mean` still wins, and a per-sector
curve cuts its CD-day error by 5.4% against a flat spread.

## What changed

- `src/forecasting/dataset.py`: `build_daily_base` (real items per day, and
  each sector's window; days without an order are recovered as zeros).
- `src/forecasting/daily.py` (new): curve (`uniform` | `sector`, in tenths of
  the window), CD split, and the CD-day / sector-day metrics.
- `evaluation.py` / `comparison.py`: optional `daily_base`; `compare()` ranks
  by CD-day when present. The `compare_forecasters` driver turns it on with a
  `daily:` config block. Results in `experiments/daily_output/`.

## Notes

- Decided with the user: rank by MAE per CD-day; keep the current training
  convention first (cycle k−1 whole). Both are recorded in the README.
- Block/sub-block never enters the forecast; the window's real dates only
  place it on the calendar. The old `shape.py`/`scoring.py` were left alone —
  they depend on `slot_start_day`, i.e. the block.
- Follow-ups: a training set restricted to already-closed cycles (open
  question for the client: how far ahead is the block calendar fixed?), and the
  weekday effect.
