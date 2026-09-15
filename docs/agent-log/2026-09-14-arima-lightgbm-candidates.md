# ARIMA/SARIMA and pooled LightGBM candidates, and a fair comparison between them

**Date:** 2026-09-14
**Related:** branch `tasso/forecasting-harness`, follows
`2026-09-14-forecasting-harness.md`

## Task

Implement the two remaining forecasting candidates from the spec's section
4.4 — ARIMA/SARIMA and LightGBM — on top of the interface and harness built
earlier the same day, and explain both models to the requester before
writing code.

## Outcome

Both candidates are implemented, tested and ranked against the naive
baseline on the real demand base. **The result contradicts the hypothesis
that was pre-registered before the run:** `naive:mean` — simply averaging a
sector's own history — beats both, at 1.851 items MAE against LightGBM's
1.900 and ARIMA's 2.397 on the 3.010 points every candidate scored. The
forecasting side now has a defensible answer to "which model produces
`q_{s,a,d}`", and it is a much simpler one than expected.

## What changed

- `src/forecasting/candidates/arima.py` (new) — SARIMAX per sector through
  the existing `per_sector` Adapter.
- `src/forecasting/candidates/lightgbm.py` (new) — gradient boosting pooled
  across all sectors, implementing `ForecastCandidate` directly.
- `src/forecasting/features.py` — filled in (was an empty stub): lag,
  rolling-mean and calendar features.
- `src/forecasting/comparison.py` (new) — common-subset ranking.
- `src/forecasting/evaluation.py` — extracted `equal_weight_mae` so the
  harness's summary and the comparison compute MAE the same way.
- `src/config/schema.py` — `LightGBMParams` gained `rolling_windows`,
  `min_child_samples` and `random_state`.
- `experiments/compare_forecasters/` + its config (new).
- `tests/forecasting/` — 31 new tests; the forecasting modules stay at 100%
  coverage and radon grade A.
- `pyproject.toml` — `lightgbm` added to the `forecast` extra.

## Notes

- **The headline finding is that the simplest candidate wins.** Both
  readings of it are recorded in the experiment README: with one year of
  history there's no seasonal signal, and per-sector trend is weak next to
  cycle-to-cycle noise. `naive:last_value` finishing 4th (2.522) while
  `naive:mean` finishes 1st (1.851) is the same fact seen from the other
  side — chasing the last cycle chases noise.
- **A methodological trap was found while running, not while designing.**
  Candidates skip different (sector, cycle, fold) points — LightGBM cannot
  predict the first fold at all (a 6-cycle rolling mean needs 6 prior
  cycles), forfeiting 606 points; some sectors don't fit an ARIMA order.
  Ranking each candidate by its own headline MAE would therefore have
  rewarded whichever one skipped the hardest points, and LightGBM's raw
  number did look better than it deserved. `comparison.py` exists to fix
  that: the ranking runs on the intersection, with each candidate's own
  coverage reported beside it.
- **Two deliberate deviations from the spec's section 4.4**, both
  documented in the code that makes them: (1) ARIMA's order comes from the
  config and is applied uniformly, not searched per sector by AIC — with
  6-11 points a per-sector order search selects noise and multiplies
  runtime by the grid size; (2) the year-ago lag feature and any seasonal
  order are unusable on a single year of data, so `seasonal_naive` and
  seasonal ARIMA orders are implemented and guarded but will skip every
  sector until a second year exists.
- **LightGBM uses its native API rather than the scikit-learn wrapper**, to
  avoid adding scikit-learn as a project dependency for one convenience
  class. It trains with `objective: regression_l1` so the model optimizes
  the same loss (MAE) the harness ranks by.
- A small hyperparameter grid showed a monotone preference for smaller,
  shorter LightGBM models — itself evidence of how little structure there
  is to fit. `num_leaves: 7` was chosen on that principle rather than by
  taking the grid's minimum, because the grid was scored on the same folds
  the result table reports; the caveat is stated in the experiment README
  rather than left implicit.
- **A robustness bug in `comparison.py` was found by a test, not by a run:**
  `EvaluationResult.summary()` refuses to aggregate an empty result (correct
  for a single candidate's headline number), which meant one candidate that
  scored nothing would sink the entire comparison. `_candidate_row` now
  reports NaN for such a candidate instead.
- Concrete next step recorded in the experiment README: fit LightGBM to the
  *deviation from each sector's mean* rather than the level. Every model
  here spends most of its capacity re-learning a sector's level, which
  `naive:mean` gets for free.
- Same environment caveat as the previous entry: no `uv` in the sandbox, so
  `uv.lock` was not regenerated after adding `lightgbm`. A human should run
  `uv lock` before merging. Verification used a scratch virtualenv on
  Python 3.14 with pandas 3.0.5, statsmodels 0.15.0 and lightgbm 4.7.0.

## Addendum (2026-09-14, same day)

The "what to try next" recommendation in this entry — fit LightGBM to the
deviation from each sector's mean, because "every model here spends most of
its capacity re-learning a sector's level" — was **wrong on its stated
reasoning**, and the experiment README has been corrected. Measuring instead
of asserting gave three numbers:

- The level accounts for only **20.8%** of the panel's variance; 79.2% is a
  sector varying from one cycle to the next. The level is not where the
  models' capacity goes.
- The lag-1 autocorrelation of deviations from a sector's own mean is
  **−0.065** — effectively zero. Deviations carry essentially no usable
  structure, so normalizing the target is unlikely to unlock anything.
  Only 10% of sectors exceed |0.5|, about what 11-point series yield by
  chance.
- An oracle knowing each sector's exact mean (using future data) reaches
  **1.658** items MAE against `naive:mean`'s 1.851. The entire headroom at
  this granularity is ~10%, and it isn't reachable.

One part of the original instinct survived: if the target is normalized, a
ratio is better supported than a subtraction, since absolute variation grows
with sector size while relative variation stays near 0.45.

**The more consequential finding** came out of the same check: the error at
the level the capacity constraint actually acts on is far smaller than the
per-sector number this experiment ranks by — **12.5% per CD against 41.0%
per sector**, since sector errors partly cancel across the ~42 sectors a CD
aggregates. The forecasting side has been selecting models on a harder
metric than the optimization requires. Adding a CD-day metric to the
comparison is now the top recommendation, ahead of any further per-sector
modeling.

## Addendum 2 (2026-09-15) — the ARIMA numbers were handicapped by a config default

Explaining the `(p, d, q)` notation surfaced a concrete bug in how the
experiment configured ARIMA, not in the candidate's code.

`ArimaParams.trend` defaults to `None`, matching statsmodels — which means
**no intercept**. For an undifferenced order (`d = 0`) on a series living
around 4,000 items, that forces the model to explain the entire level
through the AR coefficient, driving it to ~1 and degenerating the fit into a
random walk. On a synthetic series centred at 5,000 the fitted coefficient
is 1.001 with `trend=None` against −0.77 with `trend="c"`, and the residual
variance falls twelve-fold.

Setting `trend: c` on the undifferenced order moved `arima(1,0,0)` from
2,619 to 2,077 MAE on its own coverage, and from **5th place to 3rd** —
ahead of `arima(1,1,1)`, which it had previously trailed. The experiment
config now sets it explicitly, with the reasoning inline, and
`build_arima`'s docstring carries the warning so the next person
configuring an order doesn't repeat it. `arima(1,1,1)` is deliberately left
without it: at `d = 1` the differencing already removes the level and a
constant becomes a drift term, which is a different modeling choice.

The headline conclusion is unchanged — `naive:mean` still wins at 1,851
against LightGBM's 1,900 — but the published comparison had been unfair to
the ARIMA family, and both the experiment README and the shared report have
been corrected.

A useful consistency check fell out of the same investigation:
`arima(0,0,0)` with an intercept is, by construction, "no AR, no
differencing, no MA, just a constant" — the `naive:mean` candidate written
in statsmodels. It scores 1,864 against `naive:mean`'s 1,840, which is the
agreement between the two implementations one would want to see.
