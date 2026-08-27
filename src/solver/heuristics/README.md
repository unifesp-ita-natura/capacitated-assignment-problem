# `src/solver/heuristics`

Placeholder for greedy/local-search/metaheuristic approaches to the same
problem `mip/` and `cpsat/` solve — no external solver license required.

Give each heuristic its own module (e.g. `greedy.py`, `local_search.py`)
with a `solve(...)` function, and convert its result into a `SolveResult`
(`src/persistence/results.py`) at the call site the same way `mip/` and
`cpsat/` do — `wall_time_seconds` and `objective` still apply even though
`status`/`termination_condition` won't come from a solver library (use
something like `"heuristic"` / `"completed"` or `"iteration_limit"`).

## Testing

Heuristics have no external license to mock — test them directly against
small fixed instances with known optimal/expected values.
