# External features and temporal models on the level

**Date:** 2026-10-19
**Related:** run `experiments/compare_forecasters/outputs/2026-10-19_features_temporais`; config `configs/experiments/compare_forecasters/teste_features_e_temporais.yaml`

## Task

Test the external features already wired into the level forecast, think up
additional features to try, and test the temporal models — all on the real
base, one shared harness, one metric (`mae_common`). Decide whether any
feature or model beats the LightGBM baseline for producing the level.

## Outcome

On the (sector, city) level panel, the external categorical features
(region `CD_RE`, sales management `CD_GV`, state `estado`) do **not** help:
every LightGBM variant that adds them lands within 0.1 MAE of the featureless
baseline. The only change that improved accuracy was the `ratio` target
(predict a correction over the sector's running mean), the best candidate at
159.6 MAE. Companion lags (orders/volumes) hurt. Among temporal models,
`cycle_factor:mean` (166.6) nearly ties the LightGBM baseline and beats ETS
and ARIMA.

## What changed

- Added `configs/experiments/compare_forecasters/teste_features_e_temporais.yaml`:
  one run with the temporal/baseline candidates (naive mean/last, arima(1,0,0)+c,
  ets(A,N,N), cycle_factor:mean3) and eight LightGBM variants (base, CD_RE,
  CD_GV, CD_RE+CD_GV, estado, companion lags, full, ratio).
- No source changes. The census/salary/unemployment data in
  `src/processamento_dados/` is still NOT wired into the forecasting panel —
  that is a separate step (loader + merge in `build_item_panel` + surface in
  `features.py`).

## Results

Run on `data/base_tratada_v2.csv`, panel 12,103 sector-city series (620
sectors) x 31 cycles, rolling origin `min_train_cycles=20, horizon=1,
expanding` (11 one-step origins), 117,265 common points. Wall time ~1h20
(ARIMA ~30 min and ETS ~15 min per-series dominate). Ranked by `mae_common`:

| # | candidate | mae_common | bias_pct_common | type |
|---|---|---|---|---|
| 1 | lgbm_ratio | 159.6 | -26.1 | LightGBM (target=ratio) |
| 2 | lgbm_base | 165.9 | -35.1 | LightGBM |
| 3 | lgbm_regiao (CD_RE) | 165.9 | -35.1 | LightGBM + feature |
| 4 | lgbm_gerencia (CD_GV) | 165.9 | -35.5 | LightGBM + feature |
| 5 | lgbm_regiao_gerencia | 165.9 | -35.5 | LightGBM + features |
| 6 | lgbm_geografia (estado) | 166.0 | -35.4 | LightGBM + feature |
| 7 | cycle_factor:mean3 | 166.6 | -5.6 | temporal |
| 8 | ets(A,N,N) | 172.7 | +6.3 | temporal |
| 9 | naive:mean | 177.0 | +13.6 | baseline |
| 10 | arima(1,0,0)+c | 179.3 | +14.0 | temporal |
| 11 | lgbm_companion | 187.6 | -22.3 | LightGBM + orders/volumes |
| 12 | lgbm_full | 189.4 | -21.8 | LightGBM + all |
| 13 | naive:last_value | 218.3 | -3.7 | baseline |

ARIMA and ETS score fewer points (`n_missing` 6,554 vs 1,752): they skip
series with too little history to fit. All LightGBM/naive variants cover the
same 122,067 own points; 1,752 series have no prediction (no history yet).

## Notes

- The categorical features add nothing because the per-series identity
  (`serie = cd_setor|estado|cidade`) already carries the geography; region,
  management and state are just coarser groupings of a key the lags already
  condition on.
- Companion lags hurt rather than help: deeper lags of orders/volumes cost
  training rows (dropna) and the extra columns added noise, not signal.
- The win is a target reparameterization (`ratio`), not a feature. Predicting
  the correction over naive:mean is easier than relearning 12k series levels.
- Almost everything under-predicts (negative bias) because of the quiet-cycle
  zero rows; `cycle_factor` has the smallest bias (-5.6%), which may matter
  for the downstream assignment model even though its MAE is mid-pack.
- Follow-up for a human: the census/salary/unemployment data are continuous,
  exogenous signals the model has never seen — unlike the categorical
  groupings tested here — so they are still worth wiring in. But they are
  per-UF (state), and the per-UF categorical (`estado`) did not help, so keep
  expectations moderate. City-level external data would be needed to vary
  within a state.
