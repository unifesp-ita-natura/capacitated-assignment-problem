# mip_warmstart_scaling

**Status:** in-progress

**Goal:** At what number of sectors does the SA-warm-started block-assignment MIP stop solving
within the time limit, for horizons of 1 to 6 cycles?

**Hypothesis:** The time-out frontier moves to fewer sectors as cycles grow, since variables
grow with sectors (`15 * S`) and constraints with sectors plus total horizon days
(see `mip_size`). The cold MIP is not run here; compare against a separate cold run to judge what the warm start buys.

**Design note:** For each cycle count (1 to 6) sectors grow 40, 80, ..., 1000 (the largest
sectors by forecast items, as in the pipeline). Each point runs SA, then one MIP solve started from SA's assignment, with the
per-solve time limit (default 10 minutes). After `--max-timeouts` (default 2) time-outs
at a cycle count, the rest of that cycle's sectors are skipped and the next cycle count
starts. A time-out is a `maxTimeLimit` or `noSolution` termination. The sweep also stops
at a cycle count once the data has fewer sectors than requested. The forecast is built
once per cycle count and reused. One seed, one run per point, so timings are noisy.

```bash
uv run --extra forecast python -m experiments.mip_warmstart_scaling.run [--time-limit 600] [--overwrite]
```

Writes `outputs/scaling.csv` (gitignored), one row per solved point with `timed_out`.

**Resuming:** each row is appended to the CSV as soon as its point is solved. Rerunning the
same command reads the CSV, skips solved points (still counting their timeouts toward the
early stop) and continues where it stopped. Use the same `--time-limit` when resuming, or
the old rows will not match; pass `--overwrite` to discard earlier results and start over.
