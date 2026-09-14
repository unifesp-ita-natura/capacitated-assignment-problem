# compare_forecasters

**Status:** confirmed — see `configs/experiments/compare_forecasters/all.yaml`
for the pre-registered goal/hypothesis/design-note block.

**Goal:** rank every forecasting candidate implemented so far — naive, ARIMA
and pooled LightGBM — on the real demand base, through one shared harness
and one shared metric, to decide which one produces `q_{s,a,d}` for the
assignment model.

**Hypothesis (written before running):** pooled LightGBM wins, because a
per-sector model has 6–11 points to estimate from while the pooled model
sees thousands of rows across all sectors.

```bash
uv run python -m experiments.compare_forecasters.run
```

Writes `outputs/comparison.csv` (the ranking table) and
`outputs/errors_by_candidate.csv` (every scored (sector, cycle, fold) row
for every candidate, for digging into where a candidate loses).

## Result (2026-09-14) — hypothesis rejected

| rank | candidate | MAE (common subset) | MAE (own coverage) | scored | skipped |
|---|---|---|---|---|---|
| 1 | **naive:mean** | **1.851** | 1.840 | 3.689 | 0 |
| 2 | lightgbm:100x7 | 1.900 | 1.892 | 3.083 | 606 |
| 3 | arima(1,1,1) | 2.397 | 2.397 | 3.010 | 679 |
| 4 | naive:last_value | 2.522 | 2.514 | 3.689 | 0 |
| 5 | arima(1,0,0) | 2.609 | 2.619 | 3.684 | 5 |

Errors are in items. The ranking uses the **common subset**: the 3.010
(sector, cycle, fold) points every candidate predicted. Candidates skip
different points, so `mae_own` is each candidate's error over its own
coverage and is *not* comparable across rows — see
`src/forecasting/comparison.py`.

**The simplest thing wins.** Averaging a sector's own history beats every
model, including the pooled one the hypothesis favored. Two readings, both
consistent with what this base actually contains:

- **There is very little to learn beyond a sector's level.** With one year
  of history there's no seasonal signal, and per-sector trends are weak
  relative to cycle-to-cycle noise. `last_value` losing badly to `mean`
  (2.522 vs 1.851) says the same thing from the other side: chasing the
  last cycle's value chases noise.
- **ARIMA is squeezed from both directions.** Fitting 2–3 coefficients on
  6–11 points is thin, and the fixed order applied to every sector can't be
  right for all of them. `arima(1,1,1)` beating `arima(1,0,0)` is consistent
  with differencing absorbing part of the level, which is most of the signal.

**LightGBM's second place is real but narrow** (1.900 vs 1.851, ~3%), and it
comes with the worst coverage of the three families that scored broadly —
it cannot predict the first fold at all, because a 6-cycle rolling mean
needs 6 prior cycles.

### Sensitivity behind the LightGBM configuration

The config uses `num_leaves: 7`, below LightGBM's default of 31, because the
pooled training table is only a few thousand rows. A small grid over the
full base (MAE on own coverage) showed a monotone preference for smaller,
shorter models:

| num_leaves | 100 rounds, lr 0.05 | 300 rounds, lr 0.1 |
|---|---|---|
| 3 | 1.887 | 1.902 |
| 7 | 1.892 | 1.926 |
| 15 | 1.904 | 1.941 |
| 31 | 1.898 | 1.957 |

That the trend is monotone rather than a lucky spike is itself evidence that
there is little structure to fit. **Caveat:** this grid was scored on the
same folds the result table reports, so those numbers are optimistic as an
estimate of unseen-data error; `num_leaves: 7` was chosen on the principle
(tiny training set → regularize hard), not by taking the grid's minimum.

## Why naive:mean wins — the series is mostly noise

Three measurements on the full panel explain the result, and bound how much
any future candidate can gain:

| measurement | value | reading |
|---|---|---|
| variance between sectors (the "level") | 20.8% | the level is a fifth of the variation |
| variance within a sector, cycle to cycle | 79.2% | the rest is a sector bouncing around |
| lag-1 autocorrelation of deviations from the sector mean | **−0.065** | being above the mean says nothing about next cycle |
| MAE of an oracle that knew each sector's exact mean | 1.658 items | uses future data — not achievable |
| MAE of `naive:mean` | 1.851 items | the honest version of the same idea |

The autocorrelation is the decisive one: deviations from a sector's own mean
carry essentially no usable structure. Only 10% of sectors exceed |0.5|,
which is roughly what 11-point series produce by chance. And the gap between
`naive:mean` (1.851) and a cheating oracle (1.658) is about 10% — that is
the entire space any better model can compete for at this granularity.

## The error that matters is at the CD, not the sector

The capacity constraint is per distribution centre per day, not per sector,
and sector errors partly cancel when summed — one sector over-orders while
another under-orders. Measured on the same `naive:mean` run:

| level the error is measured at | relative error |
|---|---|
| per sector (what this experiment ranks on) | 41.0% |
| per CD (what constraint 3.5.4 acts on) | **12.5%** |

With a median of ~42 sectors per CD, aggregation cuts relative error to
under a third. This experiment therefore ranks candidates on a harder
metric than the optimization actually requires. (Approximate: each sector is
assigned to its dominant CD, and 429 sectors appear under more than one.)

## What to try next

- **Add a CD-day error metric and rank on it too.** It's the level the MIP's
  capacity constraint acts on, and it may not pick the same winner.
- **Shorten the rolling window** so LightGBM can score the first fold and
  stop forfeiting 606 points.
- **Predicting the deviation from the sector mean** (fitting the model to
  `items / sector mean` rather than to the level) is cheap to try, but the
  autocorrelation above says there is little there to find. A ratio is
  better supported than a subtraction: absolute variation grows with sector
  size (1.609 → 2.400 items from smallest to largest quartile) while
  relative variation stays near 0.45.
- **Revisit once a second year of history exists.** Seasonal orders, the
  `seasonal_naive` strategy, and the year-ago lag feature from the spec's
  section 4.4 are all unusable today and are the main untested source of
  signal.
- **Per-sector order selection for ARIMA** (spec section 4.4) was
  deliberately not implemented — see `build_arima`'s docstring.
