# New level forecasting models

Rolling-origin backtest (11 origins, horizon 1, 6,736 sector-cycle points)
of the candidates added in `src/forecasting/candidates/` (`intermittent`,
`pretrained`, `neural`, `ensemble`, and the LightGBM `objective` option).
MAE is in items per sector-cycle, with every candidate scored on the same
points.

```bash
uv sync --extra neural
# sector level (~50 min on 8 CPU cores)
uv run python -m experiments.compare_forecasters.run configs/experiments/new_models/sector.yaml
# fixed-alpha ETS and the TSB/TimesFM/LightGBM ensembles (~1h40)
uv run python -m experiments.compare_forecasters.run configs/experiments/new_models/next_steps.yaml
# city series summed back to sector, beside the sector forecast (~20 min)
uv run python -m experiments.sector_vs_city.run configs/experiments/new_models/city.yaml
# TabPFN-TS: needs TABPFN_TOKEN (accept the license at ux.priorlabs.ai first)
uv run python -m experiments.compare_forecasters.run configs/experiments/new_models/tabpfn.yaml
```

## Sector level

"Wins" is the number of origins where the candidate's MAE is below that
of pooled-alpha ETS, the previous best.

| candidate | MAE | bias | wins |
|---|---|---|---|
| ensemble:nnls(tsb+timesfm+lgbm) | 1767.9 | −5.9% | 9/11 |
| ensemble:mean(all5) | 1784.5 | −0.6% | **11/11** |
| ensemble:mean(tsb+timesfm+lgbm) | 1785.1 | +0.9% | 10/11 |
| ensemble:mean(ets+lgbm+chronos) | 1785.7 | −0.2% | 10/11 |
| ensemble:mlp(ets+lgbm+chronos) | 1796.7 | −13% | 8/11 |
| tsb 0.1/0.1 (= adida) | 1810.1 | +2.5% | 7/11 |
| ets alpha=0.05 | 1812.6 | −2.0% | 6/11 |
| ets alpha=0.1 | 1814.6 | +2.3% | 8/11 |
| timesfm2.5 median | 1828.1 | −4.9% | 4/11 |
| ets alpha=0.15 | 1828.6 | +1.7% | 7/11 |
| ets pooled alpha | 1829.4 | +0.7% | — |
| ets alpha=0.2 | 1839.4 | +0.1% | 3/11 |
| chronos2 median | 1845.8 | −6.1% | 3/11 |
| lightgbm l1 | 1860.6 | +5.0% | 5/11 |
| lstm | 1863.3 | | 3/11 |
| chronos2 mean (decile average) | 1863.8 | | |
| ets alpha=0.3 | 1876.3 | −1.9% | 3/11 |
| mlp | 1876.9 | | |
| naive mean | 1907.6 | | |
| lightgbm l2 / tweedie | 1975.7 / 1977.1 | | |
| nhits | 1980.4 | | |

- Combining forecasts is the only change that wins clearly. The plain mean
  of five members (ets, tsb, lgbm, timesfm, chronos) beats ETS at every
  origin and is almost unbiased. nnls has the lowest MAE, but it is
  biased (−5.9%) and loses one origin by 52 items.
- No single new model clearly beats ETS. TSB with alpha 0.1 is equal to
  ETS with a fixed alpha of about 0.1, which is expected, because at sector
  level there are no zero cycles and TSB reduces to SES. A low fixed alpha
  (0.05 to 0.1) beats the alpha fitted on the pooled data, so that fit
  overreacts to recent cycles.
- Of the foundation models, TimesFM-2.5 used zero-shot is as good as ETS
  and Chronos-2 is slightly worse. Both under-forecast by 5–6%.
- The neural networks trained here (LSTM, MLP, N-HiTS) and the LightGBM
  L2/Tweedie objectives lose to ETS. Roughly 600 short series are too
  little data to train them.

## City series summed to sector

| candidate | sector | cities→sector | cities→sector bias |
|---|---|---|---|
| tsb | 1810.1 | 1885.3 | +7.4% |
| adida | 1810.1 | 1882.1 | +6.5% |
| ets pooled | 1829.4 | 1855.5 | +2.4% |
| lightgbm l1 | 1860.6 | 2021.0 | −27.5% |
| lightgbm tweedie | 1977.1 | 1895.1 | −0.6% |
| lightgbm l2 | 1975.7 | 2040.3 | +8.7% |

Forecasting the sector directly beats summing city forecasts for every
candidate. Tweedie removes the bias that L1 shows on the sparse city series,
but the summed forecast still trails sector-level ETS.
