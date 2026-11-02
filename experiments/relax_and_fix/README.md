# relax_and_fix

**Status:** confirmed (synthetic instances, one seed per size; see the limits below).

## Goal

When does Relax-and-Fix + Fix-and-Optimize (`src/solver/matheuristics/`) beat
the exact MIP on the sector -> Bloco/Subloco assignment, in solution quality
and in time, with HiGHS (the open-source solver the project must use) and with
Gurobi (as a reference only)?

## Hypothesis

Written before the runs. Up to ~80 sectors the MIP closes in seconds and the
matheuristic only ties it. Near the real size (~800 sectors) the MIP stops
closing within the time limit, and then the matheuristic returns a feasible
solution better than the MIP's incumbent. Larger windows (`W`) give better
solutions and slower subproblems; Fix-and-Optimize recovers part of what the
construction leaves on the table.

## Design note

- Instances: synthetic, from `src/generate` through the forecasting pipeline
  used by `sa_calibration` (`build_instance`), seed 7, 200/400/800 sectors, 9
  CDs. Capacity is `loose` (multiplier 1.0) or `tight`: multiplier =
  `tight_headroom` x the As-Is peak load/capacity at multiplier 1.0, so with
  `tight_headroom` 0.95 the As-Is exceeds the worst CD by ~5%.
- Reference: the full MIP (`build_block_assignment_model`) with
  `mip_rel_gap` 1e-4 and a time limit (600 s, 900 s for 7 cycles). Every
  solution, MIP or not, is re-scored by the same evaluator
  (`src/solver/matheuristics/evaluation.py`), so objective and feasibility are
  comparable.
- Matheuristic: windows of `W` groups, step `P`. A group is a CD
  (`max_group_sectors: null`) or a slice of at most N sectors of a CD. 2
  Fix-and-Optimize passes, 60-120 s per subproblem, backtracking up to 2
  (`max_backtracks`, `B0` in a method name means it was turned off).
- Configs, one per question:

| Config | Question |
|---|---|
| `smoke.yaml` | does the driver work (40 sectors) |
| `full.yaml` | 200/400/800 sectors, capacity loose or ~5% tight, 2 cycles |
| `horizon7.yaml` | 800 sectors with the real horizon of 7 cycles |
| `binding.yaml` | capacity that really restricts the optimum (As-Is 20-35% over) |

Run (the second form swaps the solver of the MIP and of the subproblems and
writes to `outputs/<name>_appsi_gurobi/`; Gurobi needs a license, see
`src/solver/README.md`):

```bash
uv run python -m experiments.relax_and_fix.run --config configs/experiments/relax_and_fix/full.yaml
uv run python -m experiments.relax_and_fix.run --config configs/experiments/relax_and_fix/full.yaml --solver appsi_gurobi
```

Limits: synthetic data; one seed per size, so differences of a few hundredths
of a percentage point are noise; windows, group size and CD order are not
calibrated; the `full` and `horizon7` runs of the first table were made
before the `max_backtracks` option existed (it does not change a run in which
no window fails, but the HiGHS `W4` row of the 7-cycle table would be rerun
with it).

## Result

Gaps are against the best MIP solution of the same solver unless noted.

### Capacity does not restrict the optimum (`full.yaml`, `horizon7.yaml`)

| Instance | MIP HiGHS | MIP Gurobi | R&F+F&O HiGHS | R&F+F&O Gurobi |
|---|---|---|---|---|
| 200 sectors, tight 5% | 19 s | 2.4 s | +0.06% to +0.08%, 36-47 s | same objective, 7-8 s |
| 400 sectors, tight 5% | 196 s | 4.6 s | +0.005% to +0.05%, 158-297 s | +0.005% to +0.015%, 17-18 s |
| 800 sectors, loose | 213 s | 8.3 s | +0.03%, 115-120 s | +0.03% to +0.05%, 28-29 s |
| 800 sectors, tight 5% | 207 s | 9.0 s | +0.03%, 149-156 s | +0.01% to +0.04%, 23-31 s |
| 800 sectors, 7 cycles | 372 s | 28 s | +0.02%, 386 s (`W3P2`, 100-sector groups) | +0.03%, 100-130 s |

In all of these the optimum is set by the churn budget, not by capacity: it is
the same with loose and tight capacity. The MIP closes everything, even at 800
sectors and 7 cycles. The matheuristic is always feasible and within 0.08% of
the optimum, but it is not faster than the MIP, except by a margin at 800
sectors with HiGHS (115-156 s against ~210 s) at a cost of ~0.03%. Fix-and-
Optimize improved the construction by 0.3% (200 sectors), 0.1% (400) and
0.02% (800).

### Capacity restricts the optimum (`binding.yaml`, 800 sectors, 2 cycles)

`tight_headroom` is how far above capacity the As-Is starts: 0.80 is ~20%
over, 0.65 is ~35% over.

| headroom | MIP Gurobi | MIP HiGHS | R&F+F&O HiGHS | R&F+F&O Gurobi |
|---|---|---|---|---|
| 0.80 | 88,794, 9 s | 88,794, 158 s | 88,826 (+0.04%), 125 s | 88,813 (+0.02%), 38 s |
| 0.75 | 88,970, 10 s | **no feasible solution in 600 s** (bound 88,940) | **89,013 feasible**, 169 s | 89,033 (+0.07%), 39 s |
| 0.70 | 90,426, 155 s | **no feasible solution in 600 s** (bound 90,185) | **90,673 feasible** (+0.27% over the Gurobi optimum), 919 s | 90,550 (+0.14%), 288 s |
| 0.65 | infeasible | infeasible | no solution (correct) | no solution (correct) |

- At 0.75 and 0.70 the HiGHS MIP hits the time limit **without any feasible
  solution**, and the matheuristic returns one, within 0.08% (0.75) and 0.54%
  (0.70) of HiGHS's own lower bound. This is the case where it pays off with
  the solver the project can ship.
- Without backtracking (`B0`) the matheuristic returns no solution at 0.70 on
  either solver: freezing the first CDs leaves a later window infeasible.
  With backtracking (default 2) it recovers a feasible solution. It does not
  change a run at 0.80 or 0.75.
- 0.65 is infeasible for the MIP too; the matheuristic reports
  `no_solution_window` and returns the As-Is.

### Conclusion

1. The implementation is correct: every solution it returns is feasible, and
   the objective matches the MIP's (the same evaluator re-scores both).
2. With Gurobi the full MIP is faster than the matheuristic in every instance
   tried, up to 800 sectors and 7 cycles.
3. With HiGHS the matheuristic is worth it only when capacity restricts the
   optimum and the MIP cannot find a feasible solution in the time limit.
   Elsewhere it ties the MIP in time and loses ~0.03%.
4. Whether real Natura data looks more like the first table or the second is
   the open question; the synthetic instances cannot answer it.

Next: run on the real demand base, calibrate `W`, `P`, group size and CD
order, give the HiGHS MIP a longer limit to see where it recovers on the
binding instances, and use the matheuristic solution as a warm start for the
MIP.
