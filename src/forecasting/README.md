# `src/forecasting`

Predicts `L_{s,k}` — total items per (sector, cycle) — the demand figure
the block/sub-block assignment MIP needs as input. See
`docs/papers/forecasting-volume-spec.tex` for the formal definition and
`docs/agent-log/` for how the target's unit and dating were decided against
the real demand base (`data/base_tratada.csv`).

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

To add a technique: create `candidates/<name>.py`, register a
`ForecastParams -> ForecastCandidate` builder under the `model` literal its
params class uses, and import the module from `candidates/__init__.py`.
Nothing in `evaluation.py` or `model.py` changes.

## Modules

- `dataset.py` — loads the raw demand base and aggregates it into the
  panels forecasting uses: `build_item_panel` (one row per sector per
  cycle: total items, dated by the cycle's opening date, block-independent)
  and `build_shape_panel` (intra-cycle distribution, for the separate shape
  problem — not consumed here). Drops cycles the base only partially
  observed (see `cycle_calendar`'s docstring).
- `model.py` — the Strategy interface, the Adapter, and the Registry/Factory
  described above.
- `metrics.py` — `mae`, `rmse`, `mase`. `PRIMARY_METRIC` fixes which one
  ranks candidates, in one place, rather than leaving it up to whichever
  candidate is being compared.
- `evaluation.py` — `RollingOriginSplit` + `evaluate()`: the one place that
  slices history by cycle, hands a candidate only the past, and scores its
  predictions against what actually happened. A candidate never sees or
  computes its own training cutoff.
- `comparison.py` — ranks several `EvaluationResult`s against each other on
  the (sector, cycle, fold) points *all* of them scored. Candidates skip
  different points, so their own headline errors aren't comparable; this is.
- `candidates/` — concrete techniques:
  - `naive.py` (last_value, mean, seasonal_naive) — the required baseline,
    and as of the `compare_forecasters` experiment still the one to beat.
  - `arima.py` — ARIMA/SARIMA per sector via statsmodels' SARIMAX, through
    the Adapter. Skips a sector whose history is too short for the requested
    order, or that SARIMAX can't fit, instead of failing the run.
  - `lightgbm.py` — gradient boosting pooled across every sector at once.
    The one candidate that implements `ForecastCandidate` directly rather
    than through `per_sector`, because it needs the whole panel.
- `features.py` — the lag / rolling-mean / calendar feature table the pooled
  candidate trains on. The rolling means are shifted by one cycle so a row
  can never see the value it is being asked to predict.

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
