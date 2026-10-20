# Shorter rolling windows give the pooled model full coverage

**Date:** 2026-09-22

## Task

Run stage 2 of `docs/plans/forecasting-feature-roadmap.md`, chosen because
stage 1's fold-level breakdown pointed straight at it: the pooled LightGBM
could not predict the first fold at all, and that same fold handed
`naive:mean` nearly its whole margin.

## Outcome

The pooled LightGBM **no longer forfeits any points**. Shortening the
rolling-mean windows from `[3, 6]` to `[2, 3]` took `n_missing` from 606 to
zero, so it now predicts all 3689 points — the same coverage as the naive
benchmark — and its common-subset error improved from 1873.2 to 1862.5
items. It still trails `naive:mean` (1834.3).

## What changed

- `src/config/schema.py` — `LightGBMParams.label`, an optional override for
  the generated candidate name. Two variants that differ only in rolling
  windows would otherwise collide in the comparison table, which keys rows
  by name.
- `src/forecasting/candidates/lightgbm.py` — `_candidate_name` honours it.
- `experiments/lightgbm_short_window/` and its config (new) — the stage 2
  experiment, pre-registered and then run.
- `docs/plans/forecasting-feature-roadmap.md`, `experiments/README.md` —
  stage 2 recorded as concluded.

## Notes

**The headline number is real but the stated mechanism is not.** The entire
common-subset gain comes from a single fold. In 202607 — the earliest fold
the long-window model could predict, trained on 571 rows — the short window
is worth 176 items. In the three middle folds, where training data is
plentiful, the 6-cycle window is *better*, by about 41 items on average. So
what the change buys is not a better feature set; it is that the model stops
being data-starved where history is thin.

**This makes the choice provisional.** Stage 6 puts a second year of history
into the panel, at which point every fold becomes data-rich — the regime
where `[3, 6]` won here. The longer window may come back, and a full
year-ago window becomes affordable for the first time. Whoever runs stage 6
should re-test the window rather than inherit `[2, 3]` as settled.

**The recovered fold is itself a loss.** 202606 scores 1926.1 against
`naive:mean`'s 1842.0. Coverage was recovered, not accuracy on that fold —
worth remembering before quoting "full coverage" as an unqualified win.
