# Forecast dashboard over archived backtest runs

**Date:** 2026-10-07
**Related:** builds on 4b91866 (backtest run archives, region/management features)

## Task

Build a simple front end to evaluate the forecasts and compare actual with
predicted per sector and cycle, using the metrics of Layza's Prophet analysis
(`feat/processamento-dados`, `previsao_prophet.py`: MAE, WMAPE, Bias per
validation window). Results must come from the run folders that the backtest
now archives (`paths.run_dir`). It also needed filters, a choice of which metric
to look at, and runs of the models on base_tratada_v2 to visualize.

## Outcome

`dashboard/index.html` opens in a browser, reads a folder of runs, and
compares actual with predicted for every model in one or more runs. It can be
sliced by sector, region, sales management, CD, state and cycle, and it shows
the metric chosen in its selector. Two runs on base_tratada_v2 exist under
`experiments/compare_forecasters/outputs/` (git-ignored) to look at.

## What changed

- `dashboard/index.html`, `dashboard/README.md`: one static page (Plotly and
  PapaParse from cdnjs). It reads `manifest.json`, `predictions.csv` and
  `comparison.csv` from each run subfolder and ignores loose legacy CSVs.
  Layza's analysis was a daily aggregate with no sector, so the page keeps her
  metrics but applies them to the harness's (sector, cycle) forecasts.
- `src/forecasting/dataset.py::sector_cycle_attributes` and
  `experiments/compare_forecasters/run.py::_execute`: `predictions.csv` now
  carries `cd_cd`, `estado`, `CD_RE` and `CD_GV` per sector-cycle, so the
  page needs no 52 MB base upload. These columns are used for slicing only and
  never as model inputs.
- `configs/experiments/compare_forecasters/runs_v2_referencias.yaml` and
  `runs_v2_organizacao.yaml`: the two runs below.
- `docs/guides/forecasting-backtest-executions.md`: documents the new
  columns and points to the dashboard.

## Notes

Results on base_tratada_v2 (20 initial cycles, horizon 1, expanding window,
6,736 common points), from each run's `comparison.csv`:

| run | candidate | mae_common | wmape | bias % |
|---|---|---|---|---|
| referencias | lgbm_regiao_gerencia | 1,796.2 | 40.9% | −5.6% |
| referencias | razao + lag 19 + media anual | 1,802.0 | 41.0% | −16.2% |
| referencias | ets(A,N,N) alpha=pooled | 1,829.4 | 41.6% | +0.7% |
| referencias | naive:mean | 1,907.6 | 43.4% | +10.7% |
| organizacao | lgbm_gerencia | 1,796.2 | 40.9% | −5.6% |
| organizacao | lgbm_regiao_gerencia | 1,796.2 | 40.9% | −5.6% |
| organizacao | lgbm_regiao | 1,798.1 | 40.9% | −5.8% |
| organizacao | lgbm_sem_organizacao | 1,800.3 | 41.0% | −5.8% |

- Region and sales management help the default LightGBM by only 0.2% (4
  items). `lgbm_gerencia` and `lgbm_regiao_gerencia` are identical because
  each of the 37 managements belongs to exactly one of the 8 regions, so the
  region adds nothing once the management is known.
- The default LightGBM with organization features ties the tuned
  `razao + lag 19 + media anual` on MAE and has a third of its bias. Nobody
  has tried the tuned configuration with `CD_GV` yet.
- About 28% of sector-cycles in v2 span several CDs (14% several states).
  The CD/state filters attribute each one to the CD carrying the most items.
- Both manifests record a dirty working tree, because this session's code was
  uncommitted when the runs executed.
- The dashboard MAE is the per-point mean (Layza's definition) and differs
  slightly from `mae_common`, which weights sectors equally.
