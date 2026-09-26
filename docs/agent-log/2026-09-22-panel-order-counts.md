# Order and volume counts reach the forecasting panel

**Date:** 2026-09-22

## Task

Two requests, in sequence. First, study the Natura e-mail thread and the
data folder the team had just received, and turn both into project context.
Second, act on stage 1 of the resulting plan: give the pooled LightGBM the
order and volume counts of earlier cycles, which the panel had been
discarding at aggregation.

## Outcome

Three things are now true that weren't before.

The **"random data" scare is resolved.** The client wrote that the extended
history from December 2024 was "dados aleatórios", which would have made a
second year of history useless. Measurement shows it means *random sample of
real orders*, not fabricated numbers: for the 643 sectors present in both
periods, a sector's mean in the old period correlates 0.858 (Spearman 0.931)
with its mean in the new one, and the monthly totals repeat the same annual
shape. The second year is usable.

The **capacity constraint has a calibration problem** that wasn't visible
before. Observed daily items per CD run at 10–30% of the limits the client
supplied, because the base is a sample — so the two scales don't compare in
absolute terms. CD 2800 exceeds its own stated limit by 45%.

The **forecasting panel now carries order and volume counts**, and the
LightGBM candidate can use their lags. This is stage 1 of a written plan for
testing further features.

## What changed

- `docs/context/natura-client-answers.md` (new) — everything the client has
  stated about business rules, CD capacity and the data, the verification of
  each claim against the files, and what's still unanswered.
- `docs/plans/forecasting-feature-roadmap.md` (new) — eight staged
  experiments, ordered by expected return, plus the evaluation protocol
  every stage has to respect and an explicit stopping criterion.
- `src/forecasting/dataset.py` — `build_item_panel` aggregates
  `total_pedidos_mascarado` and `total_volumes_mascarado` alongside items;
  both are now required columns of the raw base.
- `src/forecasting/features.py` — a `companion_lags` parameter adds lagged
  order counts, volume counts and the items-per-order ratio. Empty by
  default, so the feature table is unchanged unless a config asks.
- `src/forecasting/candidates/lightgbm.py` — carries the extra series
  through the recursive forecast, leaves the target cycle's counts unknown,
  and marks the variant in the candidate name (`+orders`).
- `src/config/schema.py` — `LightGBMParams.companion_lags`.
- `experiments/panel_order_counts/` and its config (new) — the stage 1
  experiment, pre-registered.
- `tests/fixtures/demand_sample.csv` — regenerated from the real base with
  the two new columns; same 491 rows, same order, real values.

## Notes

**Everything was eventually run.** The session started with no `uv`, no
`pip` and no pandas, so the code was first written and reviewed by reading
only. `uv` was then installed and `uv sync --extra forecast` brought the
environment up: 260 tests pass, ruff format/check are clean, and radon keeps
every touched function at grade A. Note that the `forecast` extra
(lightgbm, statsmodels) is not part of the default sync — without it, the
ARIMA and LightGBM candidate tests fail on import.

**The experiment ran and the hypothesis was half wrong, in a useful way.**
The companion lags cut the pooled model's error from 1892.0 to 1873.2 items
(−1.0%), improving in 4 of 5 folds, but it still loses to `naive:mean`
(1834.3). The pre-registered check is what earned its keep: a naive
forecaster built on orders (mean orders × mean items-per-order) scores
1841.4 and is *worse* than the mean of items, so the stated mechanism —
"orders measure the sector level more cleanly" — is not what produced the
gain. An ablation shows no single series is responsible: orders, volumes and
the ratio all land within 4 items of each other, while items-only sits
19–23 behind. The likely explanation, testable at stage 4, is that a
7-leaf model is capacity-starved and extra correlated columns supply split
points rather than information.

**Two fold-level findings worth carrying forward.** The pooled model beats
`naive:mean` on 2 of 5 folds, including the hardest cycle in the set — the
one where sectors deviate most from their own means, which is what stage 5's
cycle factor is meant to capture. And almost all of `naive:mean`'s headline
margin comes from the single first fold, which is exactly the fold the
6-cycle rolling window forfeits; that makes stage 2 more valuable than its
trivial cost suggests.

**Why the roadmap's 5-seed criterion was not followed in stage 1.** The
current LightGBM configuration has no stochastic component —
`feature_fraction` and `bagging_fraction` sit at 1.0 and no bagging is on —
so every seed trains an identical model. Running five of them would
manufacture confidence rather than measure it. Seeds become meaningful at
stage 4, where subsampling is introduced.

**Measurements that motivated the stage and are worth keeping.** On 6,705
(sector, cycle → next cycle) pairs, the previous cycle's order count
correlates 0.230 (Spearman) with next cycle's items, against 0.180 for the
previous cycle's items. But after dividing by the sector's own mean both
fall to −0.08, so the expected gain is in estimating the level with less
noise, not in predicting movement. Also measured: the pooled model trains on
**zero rows** in the first fold (the 6-cycle rolling mean needs 6 prior
cycles) and at most 2,857 rows in the last, which is well below the "~4k
pooled rows" the `compare_forecasters` README claims.

**A feature that probably hurts, left alone for now.** `cycle_number` and
`opening_month` are constant within a cycle and increase monotonically, so
with an expanding window the target's value is always outside the training
range — and a tree cannot extrapolate. They can at best encode "the most
recent cycle". Stage 5 of the roadmap is where this gets addressed.
