# Forecasting Strategy interface, evaluation harness, and naive baseline

**Date:** 2026-09-14
**Related:** branch `tasso/forecasting-harness`

## Task

Before trying LightGBM and SARIMA/ARIMA, build the plumbing every candidate
needs to share: a common `fit_predict` interface, a single evaluation
harness that performs the temporal split and computes error, and at least
one candidate running end-to-end against a real sector with a real error
number.

## Outcome

`src/forecasting/` now has a working, tested forecasting pipeline: a loader
that turns the real demand base into a per-(sector, cycle) item panel, a
Strategy/Adapter/Registry-pattern interface any technique plugs into
without touching the harness, a rolling-origin evaluation harness with an
explicit leakage guard, and the naive candidate running against
`data/base_tratada.csv` (623 sectors, MAE ≈ 2,513 items, MASE ≈ 1.06 — see
`experiments/forecast_baseline/README.md`).

## What changed

- `src/forecasting/dataset.py` (new) — `load_demand_base`,
  `cycle_calendar`, `build_item_panel`, `build_shape_panel`.
- `src/forecasting/model.py` — `ForecastCandidate` (Strategy interface),
  `per_sector`/`_PerSectorCandidate` (Adapter), `CandidateRegistry`
  (Registry/Factory) with a module-level `REGISTRY` singleton,
  `InsufficientHistoryError`.
- `src/forecasting/metrics.py` (new) — `mae`, `rmse`, `mase`, and
  `PRIMARY_METRIC` (hardcoded to `"mae"`).
- `src/forecasting/evaluation.py` (new) — `RollingOriginSplit`, `evaluate`,
  `EvaluationResult`/`OriginResult`.
- `src/forecasting/candidates/naive.py` + `candidates/__init__.py` (new) —
  the naive candidate (`last_value`, `mean`, `seasonal_naive`), registered
  into `REGISTRY`.
- `src/forecasting/README.md` — filled in (was an empty stub).
- `experiments/forecast_baseline/` + `configs/experiments/forecast_baseline/naive.yaml`
  (new) — the end-to-end run against the real base.
- `tests/forecasting/` (new, 42 tests, 100% coverage on the new modules) +
  `tests/fixtures/demand_sample.csv` (new, a 3-sector slice of the real
  base's columns).
- `pyproject.toml` — added `pandas>=2.3.0` to the `forecast` extra.

## Notes

- **Target definition changed mid-session from what
  `docs/papers/forecasting-volume-spec.tex` says.** The team settled (over
  chat, relayed by the user) that the forecast target is **items**
  (`total_itens_mascarado`, matching the CD capacity unit from Natura's
  email), not volume as the spec's 2026-09-11 addendum concluded — see
  `memory/forecasting-target-definition.md`. The spec file itself was not
  updated in this session; it needs a follow-up pass.
- **Two data problems found and handled, not just assumed:** (1) the raw
  base's `Dt Abertura` is block-dependent (11 distinct opening dates within
  a single cycle in the fixture alone), so `cycle_calendar` derives one
  opening date per cycle as `min(Dt Abertura)` across all blocks —
  block-independent by construction, not by convention; (2) the base's
  final cycles are exports-in-progress, not real demand drops — a cycle is
  dropped only if its full opening-to-closing window falls outside the
  base's observed date range, a data-driven rule rather than a hardcoded
  cycle-number cutoff. At this base's export date that rule drops 202613
  and 202614; 202601, despite being the lowest-volume cycle, is kept
  because its window is fully observed.
- **A real bug surfaced only when running against the full base, not the
  synthetic tests:** `_PerSectorCandidate.fit_predict` crashed with
  `zip() argument 2 is longer than argument 1` because not every sector
  orders in every cycle (605–625 of 633 sectors per cycle), so a sector
  present in training history can have zero rows in a given target cycle.
  Fixed by skipping a sector with no target rows before calling the
  per-sector function at all, and relaxing the zip to `strict=False`
  (sector_targets is always ≤ horizon by construction, never more). Caught
  by re-running the experiment end-to-end, not by the unit test suite —
  the synthetic test panels didn't happen to include a sector missing from
  a target cycle; a regression test for this exact scenario was added
  afterward.
- One design decision made unprompted, flagged for review: **naive is the
  only candidate shipped this session**, per the task's own priority order
  (interface → harness → one working candidate). The spec's own ordering
  argument (single year of history → no seasonal signal → LightGBM-style
  pooling likely to beat per-sector ARIMA before ARIMA is even tried) is
  recorded in `docs/agent-log` chat context but not yet reflected as an
  updated recommendation in the spec `.tex` file itself.
- MASE is `nan` for any rolling-origin fold whose training window has no
  sector with 2+ cycles of history yet (e.g. `min_train_cycles=1`) — MAE
  and RMSE stay valid regardless, since they don't depend on the naive
  in-sample denominator. Documented in
  `evaluation._naive_in_sample_mae`'s docstring and covered by a test
  rather than silently left as an edge case.
- Not done: no CI/dependency lock update (`uv.lock`) was regenerated in
  this session — no `uv` binary was available in the sandbox running the
  agent; `pandas` was added to `pyproject.toml`'s `forecast` extra by hand
  and verified against a scratch virtualenv (`pip install pandas pytest
  ruff radon`), not `uv sync`. A human should run `uv lock` before merging.
