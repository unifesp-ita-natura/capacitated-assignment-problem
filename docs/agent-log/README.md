# Agent Log

Record of what AI agents did in this repo. See
`docs/conventions/agent-log.md` for the entry format and naming convention.

## Index

- [2026-10-06 backtest execution archives](2026-10-06-backtest-execution-records.md) — User-selected run folders with CSVs, YAML snapshots, effective candidate metadata and execution status
- [2026-10-06 level feature reference](2026-10-06-level-feature-reference.md) — Complete Portuguese catalog of implemented predictors, YAML options and multi-step behavior
- [2026-10-06 organization level features](2026-10-06-level-organization-features.md) — Optional categorical region and sales-management predictors with historical assignment carry-forward
- [2026-10-06 level backtest diagnostics](2026-10-06-level-backtest-metrics.md) — Common-subset Bias, WMAPE, RMSE, MASE, P90 and worst-fold MAE alongside the unchanged level ranking
- [2026-10-05 level forecasting guides](2026-10-05-level-forecasting-guides.md) — Portuguese instructions for future level predictions, inputs, backtesting, feature experiments and final holdout

- [2026-10-04 every forecasting candidate answers (sector, cycle, opening date)](2026-10-04-every-candidate-reads-the-window.md) — `opening_scenarios` builds one scenario per day the cycle could open, and every model answers each one through an opening-day factor and a window-length law; the data barely measures the opening day, so scenarios vary about ±5% (ETS) and ±0.5% (LightGBM)
- [2026-09-28 scenario queries in the forecasting harness, on base_tratada_v2](2026-09-28-scenario-harness-v2.md) — f(sector, opening day, cycle length) through `model.forecast`, the v2 fan-out rows dropped, the v2 reference scores, and a window-aware ETS and LightGBM (small gains, all on short windows; the opening day itself moves nothing)
- [2026-09-26 ETS as a level forecasting candidate](2026-09-26-ets-level-candidate.md) — exponential smoothing implemented; on two years it beats naive:mean by 4.4% and trails LightGBM by 7.0%; on one year it loses per cycle but wins per CD-day through lower bias
- [2026-09-26 daily output for the forecasting harness](2026-09-26-daily-output-harness.md) — cycle forecasts are spread over each window's days and scored per CD-day; the candidate ranking holds, and a per-sector curve cuts CD-day error 5.4%
- [2026-09-22 roadmap stages 3 to 6](2026-09-22-roadmap-stages-3-to-6.md) — the pooled model finally beats naive:mean on both panels; early stopping was the single biggest gain, the shared cycle factor is rejected as unpredictable, and the year-ago lag pays off
- [2026-09-22 shorter rolling windows give the pooled model full coverage](2026-09-22-lightgbm-short-window.md) — stage 2 run: n_missing 606 -> 0 and a small accuracy gain, but the gain comes entirely from the most data-starved fold
- [2026-09-22 order and volume counts reach the forecasting panel](2026-09-22-panel-order-counts.md) — client thread and data folder verified into project context, an eight-stage feature roadmap written, and its stage 1 run: the order counts help the model by 1% but not for the predicted reason, and still lose to naive:mean
- [2026-09-14 ARIMA and LightGBM candidates](2026-09-14-arima-lightgbm-candidates.md) — the two remaining spec candidates implemented and ranked; the naive mean baseline beat both
- [2026-09-14 forecasting harness](2026-09-14-forecasting-harness.md) — Strategy/Adapter/Registry candidate interface, rolling-origin evaluation harness, and naive baseline running end-to-end against the real demand base
- [2026-09-08 forecasting volume spec](2026-09-08-forecasting-volume-spec.md) — literature study and Overleaf-bound $L_{i,c}$ mathematical definition, model candidates, and comparison criteria
