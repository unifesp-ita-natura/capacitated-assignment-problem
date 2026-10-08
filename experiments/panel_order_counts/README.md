# panel_order_counts

**Status:** confirmed — hypothesis partly upheld: the order counts help the
model, but not for the reason predicted, and not enough to beat the
benchmark.

Stage 1 of `docs/plans/forecasting-feature-roadmap.md`.

## Goal

Does giving the pooled LightGBM the **order and volume counts of earlier
cycles** — series the panel used to throw away at aggregation — lower its
error against the `naive:mean` benchmark it currently loses to?

## Hypothesis (written before running)

It helps. Measured on 6,705 (sector, cycle → next cycle) pairs of the real
base:

| preditor do ciclo anterior | Pearson | Spearman |
|---|---|---|
| `itens_{k-1}` — what the model uses today | 0,151 | 0,180 |
| `pedidos_{k-1}` — discarded at aggregation | 0,215 | **0,230** |
| `volumes_{k-1}` — discarded at aggregation | 0,219 | — |

An order is roughly "a consultant who bought at all", and a sector's active
consultant base moves slowly. Items per order is volatile — it swings with
the magazine, the promotion and the cart size. The item count mixes the two,
so the panel was discarding the cleaner of the two signals.

**Caveat registered before running:** after dividing by the sector's own
mean, the correlation falls to **−0.08** for both series. So the expected
gain is in *estimating the level with less noise*, not in predicting
movement. If a gain appears, it should also appear in a naive forecaster
built on orders (mean of orders × items per order), which is not implemented
here — if it appears only for LightGBM, the explanation above is wrong and
the result needs a different one.

## Design note

**What varies:** `LightGBMParams.companion_lags`, empty → `[1, 2, 3, 4]`.
That switch adds 12 columns to the feature table: `orders_lag_1..4`,
`volumes_lag_1..4` and `items_per_order_lag_1..4`. The ratio is handed over
precomputed because a tree splits one column at a time and can never form
`items / orders` on its own.

**What is held fixed:** split (`horizon=1`, `min_train_cycles=6`,
expanding), target (`items`, absolute level), objective (`regression_l1`),
and every hyperparameter — `num_leaves: 7`, `n_estimators: 100`,
`learning_rate: 0.05`, `min_child_samples: 5`. The two LightGBM rows differ
in nothing but the switch.

**Why coverage can't confound it:** the companion lags reach exactly as far
back as the item lags (4 cycles), so both rows drop the same incomplete
training rows and predict the same points. A difference in MAE is a
difference in accuracy, not in which points each row chose to answer.

**Benchmarks carried along:** `naive:mean` (the standing winner, MAE 1.851)
and `naive:last_value` (the floor).

### Two known limitations

**One seed, and that is currently the correct choice.** The roadmap's stage 1
asks for agreement across 5 seeds. That criterion is not applicable yet:
this configuration has **no stochastic component** — `feature_fraction` and
`bagging_fraction` sit at LightGBM's default of 1.0 and no bagging is
enabled — so every seed trains an identical model. Seeds become informative
only in stage 4, when subsampling is introduced. Running 5 identical fits
now would manufacture false confidence.

**The folds are not fresh.** These are the same folds earlier experiments
selected `num_leaves` on, so any number here is optimistic as an estimate of
unseen-data error. Stage 0 of the roadmap (a held-out final window) is what
fixes this, and no result from this experiment should go in the paper as a
headline number before that exists.

## How to run

```bash
uv run python -m experiments.panel_order_counts.run
```

Writes `outputs/comparison.csv` (the ranking) and
`outputs/errors_by_candidate.csv` (every scored (sector, cycle, fold) row per
candidate).

## Result (2026-09-22) — the feature helps, the explanation does not

All errors are in **items**, with no thousands separator. (Earlier READMEs in
this folder write the same magnitudes as "1.851", using `.` as a thousands
separator — read those as 1851 items, not 1.851.)

| rank | candidate | MAE (common subset) | MAE (own) | scored | skipped |
|---|---|---|---|---|---|
| 1 | **naive:mean** | **1834.3** | 1840.0 | 3083 | 0 |
| 2 | lightgbm:100x7+orders | 1873.2 | 1873.2 | 3083 | 606 |
| 3 | lightgbm:100x7 | 1892.0 | 1892.0 | 3083 | 606 |
| 4 | naive:last_value | 2503.6 | 2513.5 | 3083 | 0 |

**The panel change works.** Adding the companion lags cuts the pooled
model's error by 18.8 items, about **1.0%**, and the improvement holds in
**4 of the 5 folds**:

| fold (origin) | naive:mean | lightgbm | lightgbm+orders | gain | vs naive:mean |
|---|---|---|---|---|---|
| 202607 | 1690.8 | 1901.8 | 1915.4 | −13.6 | +224.6 |
| 202608 | 1672.6 | 1719.1 | 1700.4 | +18.7 | +27.8 |
| 202609 | 2309.8 | 2270.2 | 2231.2 | +39.0 | **−78.6** |
| 202610 | 1829.6 | 1813.3 | 1795.8 | +17.5 | **−33.8** |
| 202611 | 1656.1 | 1713.8 | 1687.6 | +26.2 | +31.5 |

The only fold that gets worse is the first one, which trains on 571 rows.
Two folds (202609, 202610) are actually **won** by the order-count model
against `naive:mean` — and 202609 is the hardest cycle in the set, the one
where every sector departs furthest from its own mean. Almost all of
`naive:mean`'s headline advantage comes from a single fold (202607, +224.6).

**The pre-registered check fails, and it matters.** The README predicted
that if the gain came from estimating the level with less noise, it would
also appear in a naive forecaster built on orders. Built and scored on the
same 3083 points:

| forecaster | MAE |
|---|---|
| `naive:mean` (mean of items) | 1834.3 |
| mean of orders × mean items-per-order | 1841.4 |

The orders-based naive forecaster is **worse**, not better. So the stated
mechanism — "orders measure the sector's level more cleanly" — is **not**
what produced the gain.

### Ablation: no single series is responsible

| variant | MAE (common subset) |
|---|---|
| orders + volumes, no ratio | 1869.1 |
| orders + ratio | 1869.2 |
| volumes only | 1873.2 |
| orders + volumes + ratio (the shipped config) | 1873.2 |
| items only (baseline) | 1892.0 |

Every variant that adds *any* companion series lands in a 4-item band
(1869–1873), while items-only sits 19–23 items behind. The gain does not
attribute to orders, to volumes, or to the ratio — it appears as soon as
correlated columns are added at all.

**The most likely reading**, which stage 4 can test: with `num_leaves: 7`
and 100 rounds, the model is capacity-starved, and extra correlated columns
give the shallow trees more usable split points rather than new information.
If that is right, the same ~1% should show up from tuning alone, and the
companion lags should earn more once the target is normalized (stage 3), or
nothing at all.

### Decision

Per the rule fixed in advance: `+orders` beats its own baseline, so **the
panel change is kept and stage 2 builds on it**. It does not beat
`naive:mean`, so nothing here is reported as beating the benchmark.

Two things this changes in the roadmap:

- **Stage 5 (the common cycle factor) gets more interesting.** The pooled
  model wins exactly on the fold where sectors deviate most from their
  means, which is what a cycle factor is meant to capture.
- **Stage 2 (short rolling window) matters more than its cost suggests.**
  The first fold is both the only fold the feature hurts and the fold that
  hands `naive:mean` nearly its whole margin — and it is the fold that a
  6-cycle window forfeits entirely.
