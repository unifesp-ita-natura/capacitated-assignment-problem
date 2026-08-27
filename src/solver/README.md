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

`mip/` requires a Gurobi license to run for real (tests mock it — see
below). Point to it with a `.env` file (gitignored):

```
LICENSE_PATH="/path/to/gurobi.lic"
GUROBI_PATH="/path/to/gurobi-install/"
```

## Testing

**Do not call a real solver directly in tests.** Mock `pyo.SolverFactory`
(or the `.solve()` call on it) for `mip/`, and the equivalent entrypoint for
`cpsat/`, instead of requiring a live license or a slow real solve — see
`tests/solver/mip/test_example.py` and `docs/conventions/testing.md`.
