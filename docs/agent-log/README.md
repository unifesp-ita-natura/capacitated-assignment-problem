# Agent Log

Record of what AI agents did in this repo. See
`docs/conventions/agent-log.md` for the entry format and naming convention.

## Index

- [2026-09-26 daily output for the forecasting harness](2026-09-26-daily-output-harness.md) — cycle forecasts are spread over each window's days and scored per CD-day; the candidate ranking holds, and a per-sector curve cuts CD-day error 5.4%
- [2026-09-14 ARIMA and LightGBM candidates](2026-09-14-arima-lightgbm-candidates.md) — the two remaining spec candidates implemented and ranked; the naive mean baseline beat both
- [2026-09-14 forecasting harness](2026-09-14-forecasting-harness.md) — Strategy/Adapter/Registry candidate interface, rolling-origin evaluation harness, and naive baseline running end-to-end against the real demand base
- [2026-09-08 forecasting volume spec](2026-09-08-forecasting-volume-spec.md) — literature study and Overleaf-bound $L_{i,c}$ mathematical definition, model candidates, and comparison criteria