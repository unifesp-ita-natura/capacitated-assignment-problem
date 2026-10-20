# ETS as a level forecasting candidate

**Date:** 2026-09-26

## Task

Implement ETS (exponential smoothing) to forecast the level `L_{s,k}`, the
total items per sector and cycle. The spec's section 4.4 listed it, but the
harness had no implementation.

## Outcome

ETS is now a registered candidate. On the two-year panel, ETS(A,N,N) beats
`naive:mean` by 4.4%, and LightGBM beats ETS by 7.0%. On the one-year panel,
ETS is 9.9% worse per sector-cycle but slightly better (0.9%) per CD-day,
because its forecasts are less biased.

## What changed

- `src/config/schema.py`: added `ETSParams`, with error, trend,
  damped_trend, seasonal and seasonal_periods.
- `src/forecasting/candidates/ets.py`: statsmodels' `ETSModel` per sector,
  through the `per_sector` adapter. It follows the ARIMA candidate's
  contract: a minimum history, fit failures turned into skipped sectors, and
  forecasts clipped at zero.
- `tests/forecasting/candidates/test_ets.py`: new tests for the candidate.
- `experiments/ets_level/`, with configs for the one-year and two-year
  panels.
- `experiments/compare_forecasters/run.py`: now imports the whole
  candidates package, so a new candidate registers without editing the
  driver.

## Notes

- The minimum history counts only the smoothing parameters (4, 6 or 8
  cycles). A first version also counted initial states, which asked for 10
  cycles for a trend model and excluded almost every fold of the one-year
  panel.
- The metric split on the one-year panel is the finding worth keeping.
  CD-day error rewards low bias across sectors, not low per-sector error.
  The next step is a per-fold, per-CD bias check.
- Follow-up in the same session: `alpha: pooled` shares one smoothing weight
  across sectors, chosen from training history at each fold. On two years it
  beats the per-sector ETS by 1.5%, with α stable at 0.2, and runs about 30
  times faster. On one year it is a wash. Starting the level at the sector's
  mean was tested and rejected: an in-sample look-ahead drags α to 0.
- Averaging ETS with LightGBM on the two-year panel almost ties LightGBM per
  cycle and cuts bias sixfold. Scoring it per CD-day needs a daily base for
  that panel.
- Seasonal ETS (m = 19) needs 38 training cycles. Neither panel has that
  many.
