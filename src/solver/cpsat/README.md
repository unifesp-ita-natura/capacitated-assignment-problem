# `src/solver/cpsat`

Placeholder for an OR-Tools CP-SAT model of the same problem `mip/` solves.

CP-SAT builds its own `ortools.sat.python.cp_model.CpModel` and solves via
`CpSolver().Solve(model)` — it does not go through Pyomo's `SolverFactory`,
so it needs its own build/solve functions here rather than reusing `mip/`.

At the boundary (wherever this gets called from, e.g.
`experiments/compare_modeling/run.py`), convert `CpSolver`'s status/objective
into a `SolveResult` (`src/persistence/results.py`) the same way `mip/` does,
so CP-SAT runs land in the same comparison table as MIP and heuristic runs.

## Dependency

CP-SAT needs the `ortools` package. Add it as an optional dependency group
in `pyproject.toml` (`[project.optional-dependencies] cpsat = [...]`) rather
than a hard dependency, since not every environment running this repo will
want it installed.

## Testing

Mock `CpModel`/`CpSolver` in tests — do not run a real CP-SAT solve in the
test suite. See `docs/conventions/testing.md`.
