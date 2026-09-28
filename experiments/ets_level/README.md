# ets_level

**Status:** confirmed on the two-year panel. On the one-year panel the
results are mixed: ETS(A,N,N) loses per cycle, wins slightly per CD-day, and
the reason is bias.

This experiment adds exponential smoothing (ETS) as a per-sector level
candidate (`src/forecasting/candidates/ets.py`). The spec (section 4.4) lists
ETS, but it had never been implemented.

```bash
uv run python -m experiments.ets_level.run configs/experiments/ets_level/one_year.yaml
uv run python -m experiments.ets_level.run configs/experiments/ets_level/two_year.yaml
```

Each config takes about 15 minutes, because statsmodels fits one model per
sector per fold.

## What ETS is, in one paragraph

ETS(A,N,N), or simple exponential smoothing, keeps one number per sector, the
"level". After each cycle it updates that level:
`level ← α·(what happened) + (1−α)·(old level)`. The forecast is the level.
α near 1 means "believe the last cycle", which is `naive:last_value`. α near
0 means "believe the long average", which is close to `naive:mean`. statsmodels
estimates α per sector.

The name is ETS(error, trend, seasonal):

- A means additive, M multiplicative, N none, and Ad an additive damped trend.
- ETS(A,A,N) is Holt's linear trend.
- ETS(A,Ad,N) is the same trend, damped so it flattens out over the horizon.

## Hypotheses (written before running)

The hypotheses are in the configs. In short:

- **One year:**
  1. ETS(A,N,N) lands within 1% of `naive:mean`.
  2. The trend variants are worse, with the damped trend between the two.
  3. ETS(M,N,N) is roughly the same as ETS(A,N,N).
- **Two years:**
  1. ETS(A,N,N) beats `naive:mean`, landing near the "mean of the last 6
     cycles" (2,677.6).
  2. LightGBM still beats ETS by more than 5%.

## Result (2026-09-26)

### One-year panel, full common subset (3,684 points, no trend variants)

| candidate | MAE per CD-day | WAPE CD-day | MAE per sector-day | MAE per cycle | mean bias per cycle |
|---|---|---|---|---|---|
| **ets(A,N,N)** | **4,538** | 35.2% | 289.1 | 2,017.1 | −150.7 |
| naive:mean | 4,579 | 35.6% | 280.0 | **1,834.0** | −436.1 |
| ets(M,N,N) | 4,582 | 35.6% | 281.1 | 1,966.6 | −388.9 |
| razão (etapa 3) | 4,620 | 35.9% | 282.5 | 1,854.6 | −354.2 |
| naive:last_value | 4,708 | 36.5% | 298.6 | 2,504.8 | +52.0 |

Bias is `forecast − actual`, so a negative value is an underforecast. The
table comes from rerunning the config without the trend variants. Their
missing folds would otherwise shrink the common subset; see the next table.

**Per cycle, hypothesis 1 is refuted.** ETS(A,N,N) is 9.9% worse than
`naive:mean` (2,017 against 1,834), not within 1%. With 6 to 11 points per
sector, the estimated α chases the most recent cycles. That makes the model
drift toward `last_value`, which is the worst row.

**Per CD-day, ETS(A,N,N) comes first, but only by 0.9%** (4,538 against
4,579). The two rankings disagree because of bias:

- Every level forecaster underforecasts on these folds, because demand grew
  across 2026.
- `naive:mean` averages in the smaller early cycles, so its bias (−436) is
  three times that of ETS (−151).
- One sector-cycle is dominated by noise, which the mean handles better.
- A CD-day sums hundreds of sectors. Their random errors cancel, and the
  shared bias does not.

For the capacity constraint, a forecaster with **lower bias** matters more
than one with lower per-sector error. That is the most useful finding here.

**Hypothesis 3 is refuted, but it does not matter much.** ETS(M,N,N) sits
between the two: better than A,N,N per cycle (1,967) and worse per CD-day
(4,582). Its multiplicative error weighs large sectors' misses less when
fitting α. Both variants skip the same 5 points, the sectors with too short a
history.

### One-year panel, with the trend variants (2,390 points: late folds only)

A damped trend needs 8 training cycles, so it skips the first folds (1,299
points missing).

| candidate | MAE per CD-day | MAE per cycle |
|---|---|---|
| ets(A,N,N) | 3,992 | 2,052 |
| naive:last_value | 4,038 | 2,557 |
| razão (etapa 3) | 4,040 | 1,881 |
| ets(M,N,N) | 4,053 | 2,005 |
| naive:mean | 4,075 | 1,895 |
| ets(A,Ad,N) | 4,084 | 2,235 |
| ets(A,A,N) | 4,124 | 2,377 |

**Hypothesis 2 is confirmed.** The undamped trend is the worst ETS variant
(+15.8% per cycle over A,N,N), and the damped trend sits between the two
(+8.9%). Less than a year of cycles is too short to separate a trend from
noise.

### Two-year panel (7,619 points, per cycle only)

| candidate | MAE per cycle (common) | mean bias |
|---|---|---|
| **razão + lag 19 + média anual** | **2,437.6** | −659.7 |
| ets(A,N,N) | 2,621.5 | +429.7 |
| ets(A,Ad,N) | 2,709.7 | — |
| naive:mean | 2,741.8 | +750.9 |

The bias column comes from the 7,640 points that `naive:mean`, ETS(A,N,N) and
LightGBM share.

**Both hypotheses are confirmed.**

- ETS(A,N,N) beats `naive:mean` by 4.4%. That is better than the "last 6
  cycles" mean measured in `two_year_panel`, whose 2,677.6 was on that
  experiment's own subset.
- Here the bias runs the other way. The panel's 2025 cycles are larger than
  its 2026 ones, so the long mean **over**forecasts, and ETS tracks the drop.
- LightGBM still beats ETS by **7.0%**. So the pooled model's win survives a
  proper recency baseline, not just the all-history mean.

## What this means

- **On the two-year panel, ETS(A,N,N) should replace `naive:mean` as the
  benchmark.** It is the fair version of "the sector's own history, weighted
  toward the recent past", and LightGBM's margin over it (7.0%) is the honest
  figure to report.
- **On the one-year panel, the headline depends on the metric.** A candidate's
  bias across sectors deserves its own column whenever the target is CD-day
  load. A next step is to measure the bias per fold and per CD. Correcting a
  shared bias, much like `cycle_factor` tried, is cheap if it is predictable.
- **Seasonal ETS is still out of reach.** With m = 19, two full seasons need
  38 training cycles, and the two-year panel has 32 in total. The candidate
  supports `seasonal`/`seasonal_periods`, but no config in this experiment
  can use them.

## Follow-up: one alpha for every sector (2026-09-26)

```bash
uv run python -m experiments.ets_level.run configs/experiments/ets_level/shared_alpha_one_year.yaml
uv run python -m experiments.ets_level.run configs/experiments/ets_level/shared_alpha_two_year.yaml
```

The per-sector fit estimates α from 6 to 11 points, so α is mostly noise.
`alpha: pooled` instead picks one α for all sectors, the one with the lowest
one-step-ahead error summed over every sector's training history. It
re-estimates that α at each fold, from training cycles only, and runs in
numpy across all sectors at once. A run takes seconds instead of minutes.

The hypotheses are in the `shared_alpha_*.yaml` configs.

### One-year panel (3,684 points)

| candidate | MAE per CD-day | MAE per cycle | bias |
|---|---|---|---|
| ets α = 0.3 (fixed) | **4,538** | 1,910 | −253 |
| ets per sector | 4,538 | 2,017 | −151 |
| **ets α = pooled** | 4,563 | 1,904 | −279 |
| ets α = 0.2 (fixed) | 4,578 | 1,862 | −495 |
| naive:mean | 4,579 | **1,840** | −436 |
| razão (etapa 3) | 4,620 | 1,858 | −354 |
| ets α = 0.1 (fixed) | 4,866 | 1,929 | −961 |

The pooled α chosen at each fold was 0.40, 0.35, 0.29, 0.28, 0.27 and 0.26.

- **Hypothesis 1 (α between 0.1 and 0.3) is partly upheld.** α sits in that
  range from the third fold on, but the first two folds, with the least
  history, chose 0.40 and 0.35.
- **Hypothesis 2 is refuted on both counts, narrowly.**
  - Per cycle, sharing α cuts the loss to `naive:mean` from 9.9% to 3.5%,
    not to under 3%.
  - Per CD-day, it lands 0.6% behind the per-sector ETS (4,563 against
    4,538). It is still 0.3% ahead of `naive:mean`.
- **Hypothesis 3 is upheld for CD-day, but not per cycle.** The fixed α that
  was best in hindsight is 0.3 per CD-day and 0.2 per cycle. The pooled
  choice (about 0.27 in the later folds) sits next to the first and 0.07 or
  more away from the second.

The fixed α = 0.1 row exposes the model's weak point: **the level starts at
the sector's first cycle.** With a small α, it takes many cycles to move
away from that start. January is a low month, so the forecast stays low,
which is where the −961 bias comes from.

I tested starting the level at the sector's training mean instead, since in
theory α → 0 should reproduce `naive:mean`. **It is worse.** The pooled α
collapses to about 0 and the model becomes `naive:mean` itself (1,840 per
cycle, 4,586 per CD-day). The reason is a look-ahead inside the training
history: a mean that includes future cycles "predicts" the history better
than any update rule. That biases the choice of α toward zero. The start at
the first cycle has no such look-ahead, so it stays.

### Two-year panel (7,640 points, per cycle)

| candidate | MAE per cycle | bias |
|---|---|---|
| razão + lag 19 + média anual | **2,423** | −660 |
| **ets α = pooled** | 2,567 | +291 |
| ets α = 0.2 (fixed) | 2,568 | +294 |
| ets α = 0.1 (fixed) | 2,570 | +391 |
| ets α = 0.3 (fixed) | 2,595 | +190 |
| ets per sector | 2,606 | +430 |
| naive:mean | 2,725 | +751 |

The pooled α was 0.18 to 0.21 in all 12 folds.

- **Both hypotheses are upheld.** The pooled ETS beats the per-sector one by
  1.5% and `naive:mean` by 5.8%. LightGBM still beats it by 5.6%.
- Here, with 20 or more cycles per sector, the pooled α is stable and matches
  the best fixed α (0.2). With enough history, the pooled α and the
  best-in-hindsight α are the same.

### Combining forecasts (from the saved `errors_*.csv`, per cycle only)

These are simple averages of forecasts already produced by the runs above, on
the points all three candidates share:

| panel | combination | MAE per cycle | bias |
|---|---|---|---|
| one year | mean(naive:mean, LightGBM) | **1,821** | −395 |
| one year | naive:mean alone | 1,834 | −436 |
| two years | LightGBM alone | **2,461** | −660 |
| two years | mean(ETS per sector, LightGBM) | 2,469 | **−115** |

On the two-year panel, averaging ETS with LightGBM almost ties LightGBM per
cycle (+0.3%) while cutting bias sixfold. Their errors lean in opposite
directions: ETS overforecasts and LightGBM underforecasts. The one-year panel
shows that CD-day error rewards low bias, so this combination is the most
promising candidate for the capacity constraint. It could not be scored per
CD-day, because the two-year panel has no daily base.

### What changes

- **On the two-year panel, pooled ETS is the recency baseline** instead of
  the per-sector one: it is better (−1.5%), about 30 times faster, and its α
  is stable at 0.2.
- **On one year, pooled ETS does not beat the per-sector one per CD-day.**
  There is not enough history for any version of ETS to stand out from
  `naive:mean`. The differences are under 1%.
- **Next step: bring the daily base to the two-year panel** and score the
  ETS + LightGBM combination per CD-day. That is where the bias reduction
  would show up.

## Caveats

- **Components are fixed per config, not selected by AIC per sector** as the
  spec describes. The reason is the one in `build_arima`'s docstring: with 6
  to 11 cycles, a per-sector search selects noise.
- The two-year panel is not scored per CD-day. Its CSV is a different extract,
  and the daily base has only been checked against `base_tratada.csv`.
- Per-cycle numbers and bias for the one-year panel come from
  `errors_one_year.csv`, restricted to the points every non-trend candidate
  scored.
