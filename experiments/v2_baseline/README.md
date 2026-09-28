# v2_baseline

**Status:** confirmed.

**Goal:** on `data/base_tratada_v2.csv`, which has two years and a daily
base, find where `naive:mean`, the pooled ETS and the two-year LightGBM land,
both per cycle and per CD-day.

```bash
uv run python -m experiments.v2_baseline.run
```

The run takes about 2 minutes.

## Why this experiment

`base_tratada_v2` is the first base that has both:

- two years of history: 620 sectors and 31 complete cycles, from Dec 2024 to
  Sep 2026;
- the daily detail that CD-day scoring needs. The two-year panel
  (`demand_two_years.csv`) was missing this.

It is also the base the scenario harness is built on
(`model.forecast`, see `src/forecasting/README.md`). All three candidates here
ignore the window, so each gives the same answer for every opening day. The
scores below are the bar that a window-aware candidate has to beat.

The loader changes two things for v2:

- **It drops the 1,190 rows that v2 repeats under several opening dates.**
  These are 163,550 items, or 0.17% of the base.
- **It rebuilds `Qtde dias`,** which v2 no longer has.

## Hypotheses (written before running)

1. Per cycle, the ranking from the two-year panel holds: LightGBM, then the
   pooled ETS, then `naive:mean`. LightGBM leads the pooled ETS by 3% to 8%.
2. Per CD-day, LightGBM's lead is smaller than per cycle. Its bias is larger
   in size than the ETS's, and CD-day error rewards low bias.

## Result (2026-09-28)

The table covers 11 folds and 6,736 points, one cycle ahead.

| candidate | MAE per CD-day | WAPE CD-day | MAE per sector-day | MAE per cycle | mean bias per cycle |
|---|---|---|---|---|---|
| **razão + lag 19 + média anual** | **5,403** | 39.4% | **267.0** | **1,814** | −619 |
| ets(A,N,N) alpha=pooled | 5,423 | 39.5% | 283.9 | 1,834 | **+8** |
| naive:mean | 5,824 | 42.5% | 295.1 | 1,903 | +420 |

Bias is forecast minus actual.

**Hypothesis 1 is partly upheld.** The ranking holds, but LightGBM leads the
pooled ETS by only 1.1% per cycle, below the 3% to 8% I expected. On
`demand_two_years.csv` the lead was 5.6%. The two bases are different
extracts, so the numbers are not comparable one to one.

**Hypothesis 2 is upheld.** Per CD-day, the lead shrinks to 0.4%. The pooled
ETS is almost unbiased (+8 items per sector-cycle). LightGBM underforecasts
by 619 on average, and that bias does not cancel when the CD-day sums
hundreds of sectors.

`naive:mean` is well behind both: 7.8% worse per CD-day than LightGBM.

### Combination (from `errors.csv`, per cycle only)

The simple average of the pooled ETS and LightGBM, on the same points:

| | MAE per cycle | bias |
|---|---|---|
| mean(pooled ETS, LightGBM) | **1,772** | −306 |
| LightGBM alone | 1,814 | −619 |

The average is 2.3% better per cycle than either model alone and halves
LightGBM's bias. The two models err in opposite directions, so their errors
partly cancel. This is the same finding as in `ets_level`, and here it wins
outright rather than tying. It has not been scored per CD-day yet; that
needs a combination candidate in the harness.

## Caveats

- One run, with LightGBM at `random_state` 0. The 0.4% CD-day gap is within
  the size of seed noise measured in earlier experiments.
- In the 11 sector-cycles that had repeated rows, the loader keeps the
  latest opening date. The owner of v2 has not yet confirmed where the
  duplication comes from.
