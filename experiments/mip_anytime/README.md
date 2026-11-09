# mip_anytime

**Status:** done

**Goal:** What does the Gurobi MIP have after 10 minutes at real size (all sectors, 6 cycles)? The
matheuristic reaches 16,995 in 9.5 min (+3.9% over the 2 h MIP); this shows whether the MIP is
already better by then.

**Hypothesis:** The MIP is within a few percent of its final value within 10 minutes, so the
matheuristic's +3.9% is not a competitive anytime result. If the cold MIP is still above 16,995
at 10 minutes, the matheuristic has a case.

**Design note:** One cold solve and one SA-warm-started solve, each with a 600 s Gurobi limit on the
pipeline's default instance. The incumbent and best bound are read from Gurobi's log (one row per
second at most) and reported at 10, 30, 60, 120, 300 and 600 s. One seed, one run per mode, so
timings are noisy. The matheuristic is not rerun here; compare against the 16,995 / 569 s figure
from the 2 h comparison.

```bash
uv run --extra forecast python -m experiments.mip_anytime.run [--time-limit 600] [--overwrite]
```

Writes `outputs/{cold,sa_warm}.json` and `.log` (gitignored). A rerun skips modes whose JSON
already exists; pass `--overwrite` to discard them.

## Result

568 sectors, 6 cycles, Gurobi, one seed. Incumbent (gap to Gurobi's best bound) by elapsed time:

| Time | Cold MIP | SA-warm MIP |
|---|---|---|
| 10 s | 16,425 (1.06%) | 16,425 (1.06%) |
| 60 s | 16,403 (0.91%) | 16,403 (0.91%) |
| 300 s | 16,395 (0.78%) | 16,395 (0.78%) |
| 600 s | 16,389 (0.73%) | 16,393 (0.77%) |

The MIP is within 1.1% of its bound after 10 seconds and 0.7% after 10 minutes. The
matheuristic's 16,995 after 9.5 min is 3.5% worse than the MIP's 10-second incumbent, so at real
size the matheuristic is not competitive with Gurobi. The SA warm start changes nothing.
Gurobi's bound (16,270) is slightly below the 2 h run's (16,283), as it has had less time.

### HiGHS (`run_highs.py`)

Same instance and 600 s budget, HiGHS instead of Gurobi, final result only (no trajectory):

| Run | Objective | Gap to bound | Time |
|---|---|---|---|
| MIP cold | 16,493 | 1.4% | 611 s (limit) |
| MIP, SA warm start | 16,511 | 1.5% | 223 s (limit) |
| Matheuristic (R&F + F&O), SA warm start | 22,762 | n/a | 114 s |

The HiGHS MIP is within 1.5% of its bound at 10 minutes, about 0.6% above Gurobi's incumbent. The
matheuristic is 38% worse than the HiGHS MIP: three subproblems hit their 60 s limit without a
solution and the run ended at 114 s of its 600 s budget. Its parameters (window, group size,
subproblem limit) are uncalibrated, so this says "does not improve as configured". The warm-started
MIP reports `maxTimeLimit` after only 223 s of wall time; the cause was not investigated, so do
not quote that timing. One seed.

```bash
uv run --extra forecast python -m experiments.mip_anytime.run_highs [--time-limit 600] [--overwrite]
```

Writes `outputs/highs_{mip_cold,mip_sa_warm,matheuristic_sa_warm}.json`.
