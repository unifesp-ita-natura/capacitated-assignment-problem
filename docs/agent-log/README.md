# Agent Log

Record of what AI agents did in this repo. See
`docs/conventions/agent-log.md` for the entry format and naming convention.

## Index

- [2026-10-30 hierarchy forecast presets](2026-10-30-hierarchy-forecast-presets.md) — Default sector-city ETS/LightGBM distribution and selectable management-city preset connected to daily shapes
- [2026-10-30 remove legacy hierarchy comparisons](2026-10-30-remove-legacy-hierarchy-comparisons.md) — Checkpoint the current implementation, then remove historical candidate execution and dedicated YAMLs while preserving retained forecasts
- [2026-10-30 standard hierarchy comparison](2026-10-30-standard-hierarchy-comparison.md) — Three default candidates with a pure five-cycle moving average and explicit historical reproduction mode
- [2026-10-30 remote main update](2026-10-30-remote-main-update.md) — Fast-forward to remote main, preserve combined forecasting and hierarchy work, and follow relocated backtest provenance helpers
- [2026-10-27 hierarchy feature ablation](2026-10-27-hierarchy-feature-ablation.md) — Controlled feature-block removal for shared distribution and reconciliation with reproduced reference forecasts
- [2026-10-27 management city history length](2026-10-27-management-city-history-length.md) — Controlled 10-versus-20 closed-cycle training comparison with explicit moving-average windows
- [2026-10-27 management city hierarchy](2026-10-27-management-city-hierarchy.md) — Configurable management-city aggregation with shared distribution, reconciliation and moving-average controls
- [2026-10-24 active city evaluation](2026-10-24-active-city-evaluation.md) — Primary city metrics conditional on positive demand, preserved full-point diagnostics and unchanged forecasts
- [2026-10-24 recent city shares](2026-10-24-city-recent-shares.md) — Optional city share features over 1, 2 and 3 closed records with a controlled hierarchy backtest comparison
- [2026-10-24 hierarchical level](2026-10-24-hierarchical-level.md) — Isolated ETS/LightGBM distribution and nonnegative WLS, moving-average baselines and strictly closed-cycle evaluation
- [2026-10-18 end-to-end pipeline](2026-10-18-end-to-end-pipeline.md) — `src.main` runs raw CSV -> combined forecast -> mip/matheuristic/sa -> As-Is vs To-Be report; As-Is slots inferred from opening dates (approximation)
- [2026-10-17 Relax-and-Fix + Fix-and-Optimize matheuristic](2026-10-17-relax-and-fix-matheuristic.md) — Windows of sector groups over the MIP, HiGHS and Gurobi; always feasible and within 0.08% of the optimum, slower than the MIP when it closes, but returns a feasible solution when the HiGHS MIP finds none (capacity-binding instances); persistent solver (3x), sector-slice windows and backtracking came from the failures
- [2026-10-16 combined daily forecast](2026-10-16-combined-daily-forecast.md) — Predicted city-cycle level × sector daily shape, isolated opening scenarios and fixed-model daily replay
- [2026-10-14 level per sector-city series](2026-10-14-level-per-sector-city.md) — Level keyed by (sector, estado, cidade) with zero rows for quiet cycles; summing cities loses to forecasting the sector for every candidate (ETS +1.4%, LightGBM +8.6% MAE)
- [2026-10-09 shape features](2026-10-09-shape-features.md) — Isolated LightGBM shape with selectable region, sales management, weekday and month; holidays deferred
- [2026-10-09 standalone shape dashboard](2026-10-09-shape-dashboard.md) — Local shape-only run filters, daily and cumulative comparisons, metrics and downloads; no level integration
- [2026-10-09 daily shape backtest](2026-10-09-daily-shape-backtest.md) — Independent sector-cycle-day curves, CD aggregation, shape-only scoring and archived runs without level changes
- [2026-10-07 forecast dashboard over backtest runs](2026-10-07-forecast-dashboard-runs.md) — Browser page comparing actual vs predicted per sector and cycle from archived runs, sliceable by region, management, CD and state; v2 runs show organization features help LightGBM by 0.2%
- [2026-10-07 simple feature-engineering manual](2026-10-07-simple-feature-engineering-manual.md) — Windows setup, a complete YAML, feature comparisons, outputs and code validation for an intern
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
