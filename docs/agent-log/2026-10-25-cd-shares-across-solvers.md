# Historical CD shares across MIP, SA and the matheuristic

**Date:** 2026-10-25
**Related:** branches `feat/cd-assignment-fixed-split`, `feat/sa-cd-shares`

## Task

Luiza asked whether a (sector, city) is served by more than one CD, then how the
solvers should deal with it. Three approaches were already on branches
(`fixed-split`, `share-variable`, `top-share`); the question was which to keep and
whether the whole pipeline honoured it.

## Outcome

In `Unifesp_Demanda_v2.csv`, 34% of (sector, city) pairs (5,415 of 15,714) have
more than one CD. The MIP, SA and the matheuristic now all weight each sector's
load across CDs by its recent historical item shares, and
`python -m src.main` builds those shares from the real data.

## What changed

- `notebooks/data_exploration.ipynb`: CDs per (sector, city), CDs vs routes, and
  CDs per cycle.
- `src/solver/inputs.py`: `effective_cd_sector_shares` (shares, or a hard 100% split
  from `cd_sectors`) shared by all three solvers.
- `src/solver/heuristics/simulated_annealing.py`: a sector-by-CD share matrix replaces
  the single `sector_cd` vector in the load, overload and feasible-move logic.
- `src/solver/real_inputs.py`: `historical_cd_shares` (items, last
  `SHARE_WINDOW_CYCLES` cycles, whole-history fallback); `sector_to_cd` is the
  top-share CD; capacity is built for every CD in the shares.
- `src/solver/matheuristics/{problem,evaluation}.py`: `sector_shares` weights the
  evaluator, which also feeds the explain stage's utilization figures.

## Notes

- Why `fixed-split`: the split is a property of the logistics network, not a
  decision of the block assignment, so `share-variable` models the wrong thing and
  needs a bilinear `x*y` term. `top-share` is the special case with shares of 0/1.
- Routes explain part of it: 41% of multi-CD pairs have disjoint route sets per CD,
  but 1,095 have a single route. Only 16% of (pair, cycle) have more than one CD, and
  48% of multi-CD pairs only switch CD between cycles.
- Relax-and-fix windows still group sectors by their top CD; the subproblems and the
  evaluator account for the full split. A near 50/50 sector lands in an arbitrary group.
- Open: the 6-cycle window is a default, not validated; CD capacities did not bind in
  the 40 and 60-sector runs (utilization about 2-4%), so capacity weighting is tested
  by unit tests but not yet stressed on real data.
