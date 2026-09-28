# `src/forecasting`

Predicts `L_{s,k}` — total items per (sector, cycle) — the demand figure
the block/sub-block assignment MIP needs as input. See
`docs/papers/forecasting-volume-spec.tex` for the formal definition and
`docs/agent-log/` for how the target's unit and dating were decided against
the real demand base (`data/base_tratada.csv`; `data/base_tratada_v2.csv`
covers two years, see `experiments/v2_baseline/`).

## How a technique plugs in

Three design patterns keep adding a technique from ever touching the
harness:

- **Strategy** — `model.ForecastCandidate` is the interface every technique
  implements: `fit_predict(history, targets) -> predictions`. The harness
  (`evaluation.evaluate`, the Strategy's "context") calls this identically
  regardless of which technique it is.
- **Adapter** — most techniques (naive, ARIMA, SARIMA) are naturally a
  function over one sector's series, not the panel-shaped `ForecastCandidate`
  interface. `model.per_sector(name, fn)` adapts that shape once; `fn`
  raises `model.InsufficientHistoryError` to say "skip this sector" rather
  than fabricate a value. A technique that pools across sectors (LightGBM —
  see the spec's section 4.5) implements `ForecastCandidate` directly
  instead, since it needs the whole panel at once.
- **Registry/Factory** — `model.REGISTRY` maps a `ForecastParams.model`
  discriminator (`src/config/schema.py`) to the builder that constructs the
  matching Strategy. Each technique registers itself from its own module
  under `candidates/` via `@REGISTRY.register("model_name")`;
  `candidates/__init__.py` imports every shipped module so registration
  runs on import.

## Scenario queries: f(sector, opening day, cycle length)

The optimizer compares opening dates, so it asks the forecast "how many
items would sector s order in a cycle of n days opening on day d?" once per
date it weighs. That question is a **query**: a `targets` row carrying
`window_start` (d) and `cycle_days` (n) next to `cd_setor` and `CICLOS`.
One (sector, cycle) can carry several queries, one per scenario.

- `model.forecast(candidate, history, queries)` answers queries from the
  whole history: the call the optimizer makes. Without `CICLOS`, every
  query is a scenario of the cycle right after the history.
- In `evaluate()`, each target is queried with the window the sector
  actually had, so the backtest scores the one scenario that happened.
- Turning (sector, d, n) into a model's own inputs (last cycles, a year
  ago, calendar variables) is the candidate's job, inside `fit_predict`;
  the history rows carry each past cycle's `window_start` and `cycle_days`
  to learn from. A candidate that returns predictions keyed only by
  (sector, cycle) ignores the window: `model.align_predictions` copies its
  value onto every scenario of that cycle. Every candidate shipped so far
  does this, so for them f is flat across opening days.

To add a technique: create `candidates/<name>.py`, register a
`ForecastParams -> ForecastCandidate` builder under the `model` literal its
params class uses, and import the module from `candidates/__init__.py`.
Nothing in `evaluation.py` or `model.py` changes.

## Modules

- `dataset.py` — loads the raw demand base and aggregates it into the
  panels forecasting uses: `build_item_panel` (one row per sector per
  cycle: total items, orders and volumes, dated by the cycle's opening
  date, block-independent — only `items` is ever a target, the other two
  are predictors through their lags)
  and `build_shape_panel` (intra-cycle distribution, for the separate shape
  problem — not consumed here). Drops cycles the base only partially
  observed (see `cycle_calendar`'s docstring). The item panel also carries
  each sector's own window (`window_start`, `cycle_days`), the scenario
  inputs. The loader drops the rows base_tratada_v2 repeats under several
  opening dates.
- `model.py` — the Strategy interface, the Adapter, the Registry/Factory
  described above, and the scenario entry point `forecast`.
- `metrics.py` — `mae`, `rmse`, `mase`. `PRIMARY_METRIC` fixes which one
  ranks candidates, in one place, rather than leaving it up to whichever
  candidate is being compared.
- `evaluation.py` — `RollingOriginSplit` + `evaluate()`: the one place that
  slices history by cycle, hands a candidate only the past, and scores its
  predictions against what actually happened. A candidate never sees or
  computes its own training cutoff.
- `daily.py` — spreads a cycle forecast over the days of the sector's window
  (`uniform` or per-sector `curve`) and over its CDs, and scores it per
  sector-day and per CD-day. Switched on by passing `dataset.build_daily_base`
  to `evaluate()`; see `experiments/daily_output/`.
- `comparison.py` — ranks several `EvaluationResult`s against each other on
  the (sector, cycle, fold) points *all* of them scored. Candidates skip
  different points, so their own headline errors aren't comparable; this is.
- `candidates/` — concrete techniques:
  - `naive.py` (last_value, mean, seasonal_naive) — the required baseline,
    and as of the `compare_forecasters` experiment still the one to beat.
  - `arima.py` — ARIMA/SARIMA per sector via statsmodels' SARIMAX, through
    the Adapter. Skips a sector whose history is too short for the requested
    order, or that SARIMAX can't fit, instead of failing the run.
  - `ets.py` — exponential smoothing (ETS) per sector via statsmodels'
    ETSModel, through the Adapter. The error/trend/seasonal components come
    from the config; the default ETS(A,N,N) is a weighted mean between
    `last_value` and `mean`. See `experiments/ets_level/`.
  - `lightgbm.py` — gradient boosting pooled across every sector at once.
    The one candidate that implements `ForecastCandidate` directly rather
    than through `per_sector`, because it needs the whole panel.
- `features.py` — the lag / rolling-mean / calendar feature table the pooled
  candidate trains on. The rolling means are shifted by one cycle so a row
  can never see the value it is being asked to predict. `companion_lags`
  optionally adds lagged order counts, volume counts and the
  items-per-order ratio; it is empty by default, so the table is unchanged
  unless a config asks for them.

## Running the baseline against the real base

```bash
# the naive baseline alone
uv run python -m experiments.forecast_baseline.run

# every candidate, ranked against each other on a common subset
uv run python -m experiments.compare_forecasters.run
```

See those experiments' READMEs for what they report — including the current
standing result, which is that `naive:mean` still beats both ARIMA and
LightGBM on this base.

## Testing

No solver mocking conventions apply here (see `docs/conventions/testing.md`
— that convention is about the MIP solver, not forecasting). Tests build
small synthetic panels directly; `tests/fixtures/demand_sample.csv` is a
3-sector slice of the real base's columns for the dataset-loading tests.
The evaluation harness tests include an explicit leakage check: a spy
candidate that records every `(history, targets)` pair it was called with,
asserting `history`'s latest opening date is strictly before `targets`'
earliest one at every origin.
