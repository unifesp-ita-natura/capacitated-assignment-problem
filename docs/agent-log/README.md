# Agent Log

Record of what AI agents did in this repo. See
`docs/conventions/agent-log.md` for the entry format and naming convention.

## Index

- [2026-09-26 daily output for the forecasting harness](2026-09-26-daily-output-harness.md) — cycle forecasts are spread over each window's days and scored per CD-day; the candidate ranking holds, and a per-sector curve cuts CD-day error 5.4%
- [2026-09-22 roadmap stages 3 to 6](2026-09-22-roadmap-stages-3-to-6.md) — the pooled model finally beats naive:mean on both panels; early stopping was the single biggest gain, the shared cycle factor is rejected as unpredictable, and the year-ago lag pays off
- [2026-09-22 shorter rolling windows give the pooled model full coverage](2026-09-22-lightgbm-short-window.md) — stage 2 run: n_missing 606 -> 0 and a small accuracy gain, but the gain comes entirely from the most data-starved fold
- [2026-09-22 order and volume counts reach the forecasting panel](2026-09-22-panel-order-counts.md) — client thread and data folder verified into project context, an eight-stage feature roadmap written, and its stage 1 run: the order counts help the model by 1% but not for the predicted reason, and still lose to naive:mean
- [2026-09-14 ARIMA and LightGBM candidates](2026-09-14-arima-lightgbm-candidates.md) — the two remaining spec candidates implemented and ranked; the naive mean baseline beat both
- [2026-09-14 forecasting harness](2026-09-14-forecasting-harness.md) — Strategy/Adapter/Registry candidate interface, rolling-origin evaluation harness, and naive baseline running end-to-end against the real demand base
- [2026-09-08 forecasting volume spec](2026-09-08-forecasting-volume-spec.md) — literature study and Overleaf-bound $L_{i,c}$ mathematical definition, model candidates, and comparison criteria