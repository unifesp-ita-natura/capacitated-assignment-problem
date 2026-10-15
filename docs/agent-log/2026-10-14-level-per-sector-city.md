# Level forecast per sector-city series

**Date:** 2026-10-14

## Task

Forecast the level (items per cycle) for each (sector, city) series instead
of each sector, in both the backtest and the scenario forecast, to study how
the city affects the forecast. Shape stays per (sector, cycle) for now.

## Outcome

`build_item_panel` returns one row per (sector, city, cycle) with a `serie`
key (`cd_setor|estado|cidade`). Every level candidate, the window-length and
opening-day adjustments, the lag features, the equal-weight MAE (one vote
per series), the common-subset comparison and the dashboard now key on
`serie`. LightGBM can also take `cd_setor`, `estado` and `cidade` as
categorical features.

## What changed

- `dataset.py`: `SERIES_KEY`, `SERIES_KEYS`, `CITY_COLUMNS`; panel grouped by
  city; `sector_cycle_attributes` no longer reports a dominant state.
- `model.py`, `features.py`, candidates (`lightgbm`, `ets`, `cycle_factor`;
  `naive`/`arima` via `per_sector`), `evaluation.py`, `comparison.py`:
  series identity is `serie`. The per-series adapter pre-groups targets
  so ~12k series don't each rescan the target frame.
- `evaluation.py`: scored rows carry `serie`, `cd_setor`, `estado`,
  `cidade`; the daily (CD-day) score sums a sector's city forecasts first.
- `schema.py`: `categorical_features` accepts `cd_setor`, `estado`, `cidade`.
- `dashboard/index.html`: common-point filter and actual totals dedupe by
  series, not by sector.
- Tests: fixture gains `cidade`/`estado` (first sector split over two
  cities); hand-built test panels use `serie`.

## Notes

A city is `(estado, cidade)`: 6,500 names map to 6,924 pairs. The base has
~12.1k series over complete cycles (620 sectors); 4,757 of 12,377
(sector, city) pairs appear in 3 cycles or fewer, and the median series
has 184 items per cycle.

The user first asked for fixed-length windows that only shift with the
opening day; data showed windows already keep the cycle's length (21 or 14
days) regardless of opening day and `opening_scenarios` already shifts
without shrinking, so the user dropped that change.

Validation: 502 tests passed, ruff clean, radon grade A. A real-base smoke
run (28 training cycles, 3 one-step origins, 14,758 common points) took
3.7 min: ets(A,N,N) pooled MAE 281.5, naive:mean 282.8, LightGBM 285.3,
LightGBM + sector/state/city categoricals 284.9. This is a smoke check, not
an archived experiment.

## Follow-up: sector vs sum of cities

`experiments/sector_vs_city` runs the same candidates and folds on a
per-sector panel (`build_item_panel(raw, by_city=False)`) and on the city
panel, sums each fold's city forecasts per (sector, cycle) and scores both
against the sector's real total (same MASE scale, common points).

Before comparing, the city panel got zero rows (`_fill_quiet_cycles`): a
city that skipped a cycle after its first purchase, while its sector was
active. Without them the backtest only asked about cities known to buy in
the target cycle, which leaks the future. The effect is large: 57.5% of the
city target rows are zeros, so a (sector, city) series buys in well under
half the cycles. The smoke numbers above predate this fix.

Run `run_c6ebfcb92ad8472998e9f6f8c6bbd26b` (data/base_tratada_v2.csv, 20
training cycles, 11 one-step origins, 6,736 common sector-cycle points,
~12 min):

| candidate | MAE [setor] | MAE [cidades→setor] | gap | bias % setor / cidades |
|---|---|---|---|---|
| ets(A,N,N) pooled | 1,829.4 | 1,855.5 | +1.4% | +0.7 / +2.4 |
| naive:mean | 1,907.6 | 1,961.4 | +2.8% | +10.7 / +13.6 |
| lightgbm 300x31, lags 1-3 | 1,860.6 | 2,021.0 | +8.6% | +5.0 / -27.5 |

Forecasting the sector directly wins for every candidate. LightGBM loses the
most: trained on L1, it predicts close to the median of an intermittent
series, so on rows where the city did buy it predicts 359 items on average
against 566 real, and the summed sector comes out 27.5% short. 1,752 city
series had no prediction (no history yet); their items count as error in
the city sum.
