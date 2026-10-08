# two_year_panel

**Status:** confirmed — the biggest win in the roadmap, and the year-ago
cycle carries real signal for the first time.

Stage 6 of `docs/plans/forecasting-feature-roadmap.md`.

## Goal

With a second year of history in the panel, does the pooled model beat
`naive:mean` — and does the year-ago cycle (lag 19) carry the seasonal
signal that one year of data could not show?

## Hypothesis (written before running)

This is the only stage that adds information rather than rearranging what
the panel already held. Every earlier stage showed the same pattern: each
change helped where training data was plentiful and hurt where it was thin.
A second year makes every fold data-rich.

## Design note

The panel is rebuilt by `experiments.two_year_panel.prepare` from the
client's v2 workbook (Dec/2024–Sep/2026) joined to the 2024–2026 cycle
calendar. **It is a different panel**: 728 sectors × 32 complete cycles and
12 folds, against 633 × 12 and 6 folds before. Error levels are therefore
**not comparable to any earlier stage** — only the within-experiment
ranking is.

The client described the extended period as *"dados aleatórios"*. That was
verified to mean a random **sample** of real orders, not fabricated numbers:
a sector's mean in the old period correlates 0.858 with its mean in the new
one (Spearman 0.931). See `docs/context/natura-client-answers.md`.

`min_train_cycles` rises to 20 so every fold can compute a year-ago lag.

## Result (2026-09-22) — hypothesis upheld, by a wide margin

| candidate | MAE (common subset) | scored | skipped |
|---|---|---|---|
| **ratio + lag 19 + annual mean** | **2468.9** | 7652 | 3 |
| ratio + lag 19 | 2473.7 | 7652 | 3 |
| ratio, no year-ago features | 2537.7 | 7652 | 3 |
| naive:mean | 2775.4 | 7652 | 3 |
| naive:seasonal_naive | 3884.7 | 7539 | 116 |

The pooled model beats `naive:mean` by **306.5 items, 11.0%** — an order of
magnitude more than anything achieved on the one-year panel, where the whole
disputed space was about 10%.

**The year-ago lag is worth 64.1 items (2.5%)** on its own (2537.7 →
2473.7), and the year-long rolling mean adds a further 4.8. This is the
first stage where a feature carries information the panel did not already
contain.

**Repeating last year's cycle outright is a disaster** (3884.7,
40% worse than the mean). So the year-ago value is useful as a *feature a
model weighs*, not as a forecast in itself — the seasonal signal is real but
weak relative to the sector's own level.

### How much of the win is just recency?

The obvious objection: with 20+ cycles of history, `naive:mean` averages
over a period whose level has shifted (2025 monthly totals run
systematically above 2026's for the same month — possibly a real decline,
possibly a different sampling rate between periods). A model that simply
weighted recent cycles more would beat it without learning anything.

Tested directly on the same points:

| forecaster | MAE |
|---|---|
| mean of all history (`naive:mean`) | 2775.4 |
| mean of the last 6 cycles | 2677.6 |
| mean of the last 3 cycles | 2788.4 |
| **pooled model** | **2468.9** |

Recency is worth 97.8 items — real, and a fairer baseline would capture it.
But the model is a further **208.7 items** beyond the best recency-only
forecaster. **About a third of the headline win is recency; two thirds is
the model.** Quoting the 11% without this decomposition would overstate the
result.

### What this settles

The pattern that ran through stages 1, 2 and 3 — every change helping where
data was plentiful and hurting where it was thin — pointed at history as the
binding constraint rather than ideas. That reading is now confirmed: given a
second year, the same model architecture goes from drawing level with the
benchmark to beating it by double digits.

### Caveats that must travel with this number

1. **Different panel, different scale.** 2468.9 cannot be compared to the
   1803.9 of stage 4. Only the rows of this table compare to each other.
2. **728 sectors, not 633.** The extended base includes sectors that were
   later rezoned out of existence; the client said to treat the calendar's
   633 as the current structure. This experiment did not filter to them.
3. **Same-folds selection.** As everywhere else in the roadmap, until stage
   0's holdout exists these numbers are optimistic.
4. **The sampling-rate question is open.** Whether the 2025-to-2026 level
   shift is real demand or a different sampling rate is still unanswered by
   the client, and it is the one open question that could change the reading.

## How to run

```bash
uv run python -m experiments.two_year_panel.prepare   # writes data/processed/demand_two_years.csv
uv run python -m experiments.two_year_panel.run
```

`prepare` reads the client workbooks from the path in the config; they live
outside git (see `data/README.md`).
