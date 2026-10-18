# `src/solver/matheuristics`

Relax-and-Fix (R&F) followed by Fix-and-Optimize (F&O) for the sector ->
Bloco/Subloco assignment problem. It is a third solver family next to the exact
MIP (`../mip/`) and Simulated Annealing (`../heuristics/`): same problem, same
inputs, and it reports into the same `SolveResult`, so the three are directly
comparable.

It reuses the Pyomo model in `../mip/block_assignment.py` unchanged and solves
it with HiGHS (`appsi_highs`), so no commercial solver is needed.

## Idea

Capacity (§3.5.4) is separable by CD, but the objective (§3.5.2-3.5.3) and the
churn budget (§3.5.5) couple every sector. So the matheuristic decomposes by
CD: every subproblem is the **full model**, but only the sectors of a few CDs
keep integer variables.

Order the CDs and slide a window of `W` groups over them. A group is a CD, or a
slice of at most `max_group_sectors` sectors of a CD: CDs are very uneven (one
CD can hold 35% of the sectors), so a window counted in CDs does not control
how many binary variables a subproblem has.

**Phase 1, Relax-and-Fix (construction).** Per iteration:

- `C_int`, the window of `W` CDs: `x[s, d]` binary.
- `C_relax`, the CDs not processed yet: `x[s, d]` in `[0, 1]`.
- `C_fix`, the CDs already processed: `x[s, d]` frozen.
- Solve, then freeze only the first `P` CDs of the window (`C_step`) and
  advance by `P`. With `P < W` consecutive windows overlap, so a CD is
  re-optimized once more before it is frozen, which helps keep feasibility.
- The last window has no relaxed CD left and freezes everything.
- If a window has no solution (infeasible, or the time limit passed with no
  incumbent), it is retried with the last `step` frozen groups released, then
  twice as many, up to `max_backtracks` times. This is what recovers a feasible
  solution when freezing the first CDs leaves a later one infeasible.

**Phase 2, Fix-and-Optimize (refinement).** Start from the feasible R&F
assignment. Release one window of CDs at a time (their `x` binary, every other
CD fixed at the incumbent), solve, and replace the incumbent only if the
true objective improves and the candidate is feasible. Sweep until a pass
changes nothing or `optimize_passes` is reached. The final solution is never
worse than the R&F one.

For `|C| = 6`, `W = 3`, `P = 2`, R&F runs three windows: `{1,2,3}` freezing
`{1,2}`, then `{3,4,5}` freezing `{3,4}`, then `{5,6}` freezing both.

## Use

```python
from src.solver.matheuristics import RelaxAndFixParams, solve

result = solve(**instance.mip_kwargs(), params=RelaxAndFixParams(window=3, step=2))
result.assignment  # {sector: combination}
result.objective  # mean over cycles of (max - min daily demand)
result.feasible  # capacity and churn respected
```

`instance.mip_kwargs()` (a `CalibrationInstance`) matches the arguments of
`build_block_assignment_model`, which `solve` takes too. `run_relax_and_fix`
wraps `solve` and returns a timed `SolveResult` (`model_name="relax_and_fix"`;
`objective` is `None` when the final solution is infeasible).

| Parameter | Default | Meaning |
|---|---|---|
| `window` (`W`) | 3 | groups with binary variables per subproblem |
| `step` (`P`) | 2 | groups frozen per R&F iteration, `1 <= P <= W` |
| `max_group_sectors` | `None` | `None`: one group per CD; `N`: slices of at most N sectors |
| `max_backtracks` | 2 | retries of a window with no solution (`0` turns it off) |
| `optimize_passes` | 2 | F&O sweeps (`0` skips phase 2) |
| `optimize_step` | 1 | stride of the F&O window |
| `subproblem_time_limit` | 60 s | per subproblem |
| `mip_rel_gap` | 1e-4 | per subproblem |

`cd_order` (an argument of `solve`) sets the order the CDs enter the windows;
the default is the order of `cd_sectors`. `result.stop_reason` is `completed` or
`no_solution_window`; in the second case the assignment is the As-Is, and the
log says whether the window was infeasible or ran out of time
(`subproblema sem solução: <termination condition>`).

## Modules

- `windows.py`: the R&F and F&O window schedules, pure functions.
- `problem.py`, `evaluation.py`: the objective, capacity excess and churn
  budget of an assignment, computed from the data (not read back from the
  solver), so feasibility never depends on solver tolerances.
- `subproblem.py`: sets each `x[s, d]` to fixed, binary or relaxed, reads the
  assignment back and wraps the Pyomo solve. The solver is persistent and
  treats fixed variables as variables, so changing a window only updates
  bounds in the solver instead of rewriting the whole model (about 3x faster).
  Works with `appsi_highs` and `appsi_gurobi`.
- `relax_and_fix.py`: the two phases and the result types.

## When it pays off

Measured in `experiments/relax_and_fix/` (synthetic instances, 800 sectors; see
its README for the tables):

- It is always feasible and within 0.08% of the optimum (0.14% to 0.27% at the
  hardest point), and it repairs an As-Is that exceeds capacity.
- When capacity does not restrict the optimum, the exact MIP closes every
  instance up to 800 sectors and 7 cycles (28 s with Gurobi, about 6 min with
  HiGHS), so the matheuristic does not beat it.
- When capacity does restrict the optimum, the HiGHS MIP found **no feasible
  solution in 600 s** on two instances, and the matheuristic returned a feasible
  one in 3 to 15 minutes. That is the case for using it with HiGHS.

## Testing

Subproblems are solved through an injectable callable
(`solve_subproblem(model, time_limit) -> bool`), so the tests in
`tests/solver/matheuristics/` drive the algorithm with a scripted fake and
never call HiGHS, per `docs/conventions/testing.md`.

## Limitations

- Every combination is valid for every CD (as in the MIP). The SA's `D_c`
  restriction is not supported here.
- R&F can still fail near the feasibility boundary: if no retry of a window
  finds a solution, `solve` returns the As-Is assignment with
  `stop_reason == "no_solution_window"` and skips F&O. More `max_backtracks`,
  overlapping windows (`P < W`) and a larger `W` all reduce it, the last at the
  cost of bigger subproblems (a window that is too big can itself run out of
  time).
- The window, group size and CD order are not calibrated yet. The defaults are a
  starting point, and only synthetic instances were tried.
- `timelimit` and `options` are APPSI keyword arguments, so `appsi_highs` and
  `appsi_gurobi` work (the gap option name is picked per solver). Another Pyomo
  solver needs its own `solve_subproblem`. `CLAUDE.md` asks not to depend on a
  commercial solver, so Gurobi is only a reference for comparisons.

## Relation to `julia/`

`julia/relax_and_fix_and_optimize.jl` (PR #8) is a reference script for a
different problem (SKU lot sizing). This package is the port of the window
logic to the real assignment formulation.
