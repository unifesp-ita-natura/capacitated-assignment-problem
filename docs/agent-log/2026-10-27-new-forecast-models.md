# New level forecasting models

**Date:** 2026-10-27

## Task

Test new model families against pooled-alpha ETS, the previous best:
- intermittent methods (TSB, ADIDA);
- zero-shot foundation models (Chronos-2, TimesFM-2.5, TabPFN-TS);
- networks trained here (LSTM, MLP, N-HiTS);
- LightGBM with L2 and Tweedie losses;
- ensembles, both averaged and learned (nnls, MLP).

Then follow up with fixed-alpha ETS and ensembles built around TSB, TimesFM
and LightGBM.

## Outcome

The plain mean of five members (ets, tsb, lgbm, timesfm, chronos) is the
most robust candidate. Its MAE is 1784.5 against 1829.4 for ETS (−2.5%),
it wins all 11 origins, and its bias is −0.6%. The nnls ensemble of tsb,
timesfm and lgbm has the lowest MAE (1767.9, −3.4%), but its bias is −5.9%
and it loses one origin. No single model clearly beats ETS. Low fixed
alphas (0.05 to 0.1) and TSB come within about 1%.

At city level, summing city forecasts back to the sector loses to the
direct sector forecast for every candidate. The Tweedie loss removes
LightGBM's −27.5% bias on the city series but does not close the gap.
Full tables are in `experiments/new_models/README.md`.

## What changed

- Candidates: `intermittent`, `pretrained`, `neural`, `ensemble`, and a
  LightGBM `objective` option (l1/l2/tweedie). `model.per_series_batch`
  sends all series in a single call.
- `pyproject.toml`: an optional `neural` extra on a CPU-only torch index.
  It pins pandas to 2.3.3.
- Configs: `configs/experiments/new_models/{sector,next_steps,city,tabpfn}.yaml`.

## Notes

- Chronos' reported "mean" is its median, so `point: mean` averages the
  deciles instead.
- TabPFN-TS needs `TABPFN_TOKEN` and was not run. Check its license before
  proposing it to the client.
- Ensembles are slow because they re-run every member at each validation
  cycle. `next_steps.yaml` took about 1h40 on 8 CPU cores.
