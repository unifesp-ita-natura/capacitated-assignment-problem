# daily_output

**Status:** confirmed on ranking, refuted on curve size (5.4% against a
pre-registered "under 5%").

Stage 0.2 of `docs/plans/forecasting-feature-roadmap.md`: score forecasts per
**CD-day**, the level the capacity constraint acts on, instead of per
(sector, cycle).

```bash
uv run python -m experiments.daily_output.run configs/experiments/daily_output/sector.yaml
uv run python -m experiments.daily_output.run configs/experiments/daily_output/uniform.yaml
```

## How a cycle forecast becomes a daily one

```
items_pred(sector, cycle)
  × the day's weight in the sector's window      (curve: uniform | sector)
  × the sector's historical share of items per CD
  = pred(sector, CD, date)
```

The curve is the sector's average share of items per **tenth of its window**
(relative position, never the block), learned from the fold's training cycles
only. The window's calendar dates are the sector's real ones and are used only
to line the forecast up with what happened — the same "drag the curve to a
slot" the MIP does. Days without an order are actuals of zero.

## Result (2026-09-26)

Same 3.689 (sector, cycle, fold) points for every row. MAE per CD-day gives
each of the 9 CDs one vote; WAPE is total CD-day error over total CD-day items.

| candidate | curve | MAE per CD-day | WAPE CD-day | MAE per sector-day |
|---|---|---|---|---|
| naive:mean | uniform | 4.847 | 37,6% | 277,9 |
| naive:mean | **sector** | **4.586** | **35,6%** | 280,0 |
| lightgbm ratio (stage 3) | sector | 4.627 | 35,9% | 282,6 |
| naive:last_value | sector | 4.712 | 36,5% | 298,6 |

1. **The candidate ranking is unchanged** by going to days: `naive:mean`,
   then the LightGBM ratio, then `last_value`, under both curves. The cycle
   total is the only thing that differs between candidates.
2. **The sector curve helps at CD-day (−5,4%) and hurts slightly at
   sector-day (+0,8%).** A curve estimated from ~9 order-days per window is
   noisy for one sector, and the noise cancels when hundreds of sectors are
   summed into a CD. That is why the ranking level matters.
3. **Daily error is large: 36% of CD-day volume.** The per-cycle relative
   error by CD in the roadmap (12,5%) is not comparable — it sums a whole
   cycle, and the day-to-day swings cancel.

## Caveats

- **Under the current convention the training set includes cycle k−1 whole**,
  whose window runs ~27 days past the opening of cycle k. So the daily
  numbers are no stricter than the per-cycle ones about what was knowable.
  Restricting to cycles already closed on the target's opening date is the
  next experiment and will worsen every number.
- **A CD-day sums only the cycles the folds scored.** Dates at the edges of
  the scored range miss contributions from unscored neighbouring cycles, in
  both actual and forecast, so the level is understated there.
- The weekday effect (weekends sell ~45% less) is not in the curve: it depends
  on which weekday the window opens, which is the block.
