# SA seed and forecast-scenario stability

**Date:** 2026-11-02

## Task

Re-solve the SA with several seeds and a few forecast scenarios and report the spread of
the objective and of the realized improvement, so the client knows whether the
recommendation is stable or a lucky draw. Done in a separate worktree, then merged.

## Outcome

New experiment `experiments/sa_seed_stability` (helpers in `src/pipeline/stability.py`).
30 runs (3 scenarios x 10 seeds): realized improvement over As-Is is 18.1% +- 2.9 pp
(min 9.3%, max 24.0%), always positive; forecast improvement is 26.5% +- 2.4. Objective CV
across seeds is 3-4.5%. Scenarios differ by less than seed noise. Details in the
experiment README.

## Decisions

- "Realized" reuses `src/pipeline/realized.py` (`realized_instance`, `rescore`) from the
  realized-demand rescore work already on this branch, instead of a second implementation;
  the numbers moved by <0.1 pp when I swapped to it.
- Scenarios: hierarchical (sector / management grouping) and naive mean.

## Open

- `AnnealingResult.time` reads 0 on this path; timings are unreliable.
- Single horizon window; no check that best-of-N seeds on forecast is best on realized.
