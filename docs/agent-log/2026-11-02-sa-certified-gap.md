# Certified SA gap at real size

**Date:** 2026-11-02
**Related:** `experiments/sa_certified_gap/`

## Task

Certify the SA's gap at real size using the best bound of a 2-hour SA-warm-started MIP run,
expecting no new compute because the logs would already exist.

## Outcome

No such log existed (the scaling CSV has only 10-minute runs and no bound; the
`docs/mip-gurobi-800-log` branch holds no log), so the run was done fresh: all 568 sectors,
6 cycles, Gurobi 12.0.3, 7200 s, started from the pipeline's default SA.

SA 37,077; MIP incumbent 16,350; best bound 16,283 (MIP gap 0.41%). The SA is certified only to
within 127.7% of optimal (56.1% of its own objective). This contradicts the 4-8% synthetic gaps.

## Notes

- Pyomo's Gurobi shell exposes the bound as `results.problem.lower_bound`; the log is kept in
  `experiments/sa_certified_gap/outputs/gurobi.log` (gitignored).
- SA ran 0.12 s with default `AnnealingParams`; a larger or calibrated SA budget was not tried,
  so the result says the pipeline's SA is far from optimal, not that SA cannot do better.
- One seed, one run.
