# `src/persistence`

`results.py` defines `SolveResult` — the common shape every model/solver
comparison should report into (model name, solver, status, termination
condition, objective, wall time, git commit) — and `write_results_csv` to
dump a list of them to disk.

Build one `SolveResult` per run regardless of which modeling approach or
solver produced it, so different formulations land in the same table and are
directly comparable. See `experiments/compare_modeling/` for the pattern in
use.

`git_commit` defaults to the current `HEAD` SHA (`current_git_commit()`,
via `git rev-parse HEAD`) unless overridden, so a result is traceable back
to the exact code that produced it — this is what backs the `confirmed`
status in `experiments/README.md`'s convention: "reproducible from the
cited config/code" is only checkable if the result says which commit ran.
It's `None` outside a git repo or without git installed, not an error.

## MLflow (optional)

`mlflow_tracking.py` logs a `SolveResult` as an MLflow run — tags (status,
termination condition, solver, git commit), params, metrics (objective,
wall time), and optional artifacts (e.g. a written `results.csv`).

- Install the extra: `uv sync --extra tracking`.
- Call `configure_tracking(tracking_uri=..., experiment_name=...)` once at
  startup, or set `MLFLOW_TRACKING_URI` in the environment — `log_solve_result`
  no-ops silently until tracking is configured, so instrumenting a run doesn't
  require MLflow to be installed or reachable.
- Logging failures are caught and warned, never raised — an unreachable
  tracking server cannot fail a solve or a comparison run.

**Do not hit a real MLflow server in tests** — mock `mlflow` itself, the same
way Gurobi is mocked. See `tests/persistence/test_mlflow_tracking.py`.
