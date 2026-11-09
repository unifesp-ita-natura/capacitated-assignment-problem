# sa_certified_gap

**Status:** done

**Goal:** Certify the SA's optimality gap at real size (all sectors, 6 cycles) using the best
bound of a long SA-warm-started Gurobi run.

**Hypothesis:** The MIP will not close within 2 h, but its best bound is a valid lower bound on
the optimum, so `(SA - bound) / bound` is a certified upper bound on how far the SA is from
optimal. So far the only gap figures (4-8%) come from synthetic instances.

**Design note:** One run, one seed. SA runs with the pipeline's default parameters and seed, then
Gurobi (`TimeLimit` 7200 s, `MemLimit` 16 GB) is warm-started from SA's assignment. The SA result
is written to `outputs/result.json` before the MIP starts, so it survives a crash. Two gaps are
reported: `vs_incumbent = (SA - LB) / SA` (comparable to Gurobi's own gap) and
`vs_optimum = (SA - LB) / LB` (the certified bound on SA's excess over the optimum). The MIP
incumbent's own gap is reported alongside.

```bash
uv run --extra forecast python -m experiments.sa_certified_gap.run [--cycles 6] [--max-sectors N] [--time-limit 7200] [--mip-gap 1e-4]
```

The MIP stops at whichever comes first: `--time-limit` seconds, or a relative gap below
`--mip-gap` (e.g. `--mip-gap 0.01` for 1%). Either way the best incumbent is returned. Gurobi
reports a gap stop as `termination: optimal`, so read `gap` next to it: it is the real distance
to the bound.

Writes `outputs/result.json`, `outputs/assignment.json` (the MIP incumbent, sector -> combination)
and `outputs/gurobi.log` (gitignored).

**Results:** (568 sectors, 6 cycles, seed 42, Gurobi 12.0.3, 7200 s)

| | objective |
|---|---|
| SA (default params, 0.12 s, feasible) | 37,077 |
| MIP incumbent after 2 h (warm-started from SA) | 16,350 |
| MIP best bound after 2 h | 16,283 |

- The MIP did not close, but its own gap is only **0.41%**.
- SA certified gap: `vs_incumbent` = 56.1%, `vs_optimum` = **127.7%**. The SA is *not* near
  optimal at real size: the optimum is at most 16,350 and the SA is 2.3x that.
- This contradicts the 4-8% synthetic-instance figures. Those gaps likely do not carry over to
  the real instance, or the default SA parameters (which the pipeline uses) are far from the
  calibrated ones. SA stopped after 0.12 s, so the search budget is a candidate cause; this run
  did not test it.
- The positive framing is the MIP's: warm-started Gurobi is within 0.41% of optimal at real size
  in 2 h. Whether it gets there much sooner (see `mip_warmstart_scaling`) is not measured here.
