# `src/solver`

Solver backends live under here as sibling families, each solving the same
class of problem a different way but reporting into the same
`SolveResult` (`src/persistence/results.py`) so they're directly comparable:

- `mip/` — exact MIP solvers via Pyomo (Gurobi, HiGHS, ...). `mip/example.py`
  shows the pattern: build a `ConcreteModel`, hand it to
  `pyo.SolverFactory(solver_name).solve(...)`. `solver_name` is any
  Pyomo-registered name, so switching backends is a config value, not a
  rewrite.
- `cpsat/` — OR-Tools CP-SAT. Not Pyomo-based — CP-SAT builds its own
  `cp_model.CpModel` and has its own solve loop — so it gets its own module
  rather than going through `mip/`. See `cpsat/README.md`.
- `heuristics/` — greedy/local-search/metaheuristic approaches that need no
  external solver license. See `heuristics/README.md`.

Add real families the same way: a subpackage with its own build/solve
functions, converting its native result into a `SolveResult` at the edge
(don't leak solver-specific result types past the module boundary — that's
what makes the comparison runner in `experiments/compare_modeling/` able to
treat every family the same).

## Gurobi license

`mip/` defaults to `solver="gurobi"`. Point to your license with a `.env`
file (gitignored):

```
LICENSE_PATH="/path/to/gurobi.lic"
GUROBI_PATH="/path/to/gurobi-install/"
```

If no working Gurobi license/install is found (checked via `.available()`,
never by letting `.solve()` fail), the default falls back to the bundled,
license-free `appsi_highs` and logs a warning — see `solve_example` in
`mip/example.py` and `solve_block_assignment`/`run_block_assignment_mip` in
`mip/block_assignment.py`. This keeps the project runnable on a machine with
no Gurobi license.

## Testing

**Prefer solving for real with HiGHS** (`solver="appsi_highs"`) over mocking
`pyo.SolverFactory` — it needs no license, so tests get real `SolverResults`
instead of asserting on a mock's call log. Mock only to simulate "Gurobi has
no license" (a small stub whose `.available()` returns `False`), leaving
every other solver name, especially the HiGHS fallback, going through the
real `pyo.SolverFactory`. See `tests/solver/mip/test_example.py`,
`tests/solver/mip/test_block_assignment.py`, and
`docs/conventions/testing.md`.
