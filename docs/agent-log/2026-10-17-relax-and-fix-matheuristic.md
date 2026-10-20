# Relax-and-Fix + Fix-and-Optimize matheuristic for the assignment problem

**Date:** 2026-10-17
**Related:** PR #8 (Julia reference script for a different problem), branch
`feat/relax-and-fix`

## Task

Mateus asked for a correct Relax-and-Fix implementation for the actual
problem (sector -> Bloco/Subloco, the MIP in `src/solver/mip/`). PR #8 only
holds a Julia reference script that solves a lot-sizing problem, and Luiza
requested changes on it. The presentation slides already describe the method:
windows of CDs, `C_fix`/`C_int`/`C_relax`/`C_step`, then Fix-and-Optimize.

## Outcome

`src/solver/matheuristics/` solves the same Pyomo model as the MIP, by windows
of sectors, with HiGHS (Gurobi also works, as a reference). Every solution it
returns is feasible and within 0.08% of the optimum on the instances tried
(0.14% to 0.27% at the hardest point). It is not faster than the exact MIP
when the MIP closes (Gurobi: 800 sectors, 7 cycles in 28 s; HiGHS: about 6
min). It wins when capacity restricts the optimum: the HiGHS MIP found no
feasible solution in 600 s on two such instances and the matheuristic returned
one. Whether real data behaves like that is still open.

## What changed

- `src/solver/matheuristics/` (new): `windows.py` (R&F and F&O schedules,
  backtracking), `problem.py` and `evaluation.py` (objective, capacity excess
  and churn budget computed from the data), `subproblem.py` (sets each `x`
  fixed, binary or relaxed; persistent Pyomo solver) and `relax_and_fix.py`
  (the two phases). 118 tests in `tests/solver/matheuristics/`, 100%
  coverage, solver mocked through an injectable callable.
- `experiments/relax_and_fix/` and `configs/experiments/relax_and_fix/`:
  driver and four configs (smoke, full, horizon7, binding), `--solver` flag.
- `src/solver/README.md`, `src/solver/__init__.py`: the new family.

## Notes

- The first version spent almost all its time in Pyomo rewriting the whole
  model for every subproblem (profile: expression visitors, `standard_repn`),
  not in HiGHS. A persistent `appsi` solver with
  `treat_fixed_vars_as_params = False` only updates variable bounds: same
  objective, about 3x faster.
- A window counted in CDs does not bound the subproblem: CDs are very uneven
  and the first window (3 CDs) held 557 of 800 sectors. It took 133 s and ran
  out of the 60 s limit with no solution. Windows now count groups, optionally
  slices of a CD (`max_group_sectors`).
- That failure was labelled `infeasible_window`, which was wrong: it was a
  time limit. The status is now `no_solution_window` and the termination
  condition goes to the log.
- Near the feasibility boundary (As-Is 30% over capacity) R&F froze the first
  CDs and left a later window infeasible. Backtracking (release the last
  frozen groups and retry) recovers a feasible solution there.
- The optimum on these synthetic instances is set by the churn budget, not by
  capacity, until capacity gets tight enough to be almost infeasible; the
  transition is sharp (headroom 0.75 binds, 0.65 is infeasible).
- The environment's `.venv` was missing `pandas` and `highspy`; `uv sync`
  fixed it. Gurobi has an academic license on this machine (non-commercial).
- Not done: real demand data, calibration of `W`, `P`, group size and CD
  order, a longer HiGHS time limit on the binding instances, and using the R&F
  solution as a warm start for the MIP. Nothing is committed yet.
