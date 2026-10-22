# mip_size

**Status:** confirmed

**Goal:** How large (variables, constraints) does the block-assignment MIP get as
sectors go from 40 to 1000 and the horizon from 1 to 20 cycles?

**Hypothesis:** Variables grow linearly in sectors (`15 * S`, the cycles add only
`2 * H`); constraints grow linearly in sectors and in total horizon days
(`S + 4 * days + 1` with 2 CDs), so cycles dominate the constraint count at
small S and sectors dominate at large S.

**Design note:** Nothing is solved or built. Counts come from the closed form in
`experiments/mip_size/model_size.py`, checked against a really built Pyomo model in
`tests/experiments/test_mip_size.py`. Instances use 15 combinations, 2 CDs and a fixed
21-day cycle, set in `run.py` (no dependency on `compare_modeling`). Sectors step by 40.

```bash
uv run python -m experiments.mip_size.run
```

Writes `outputs/sizes.csv` and `outputs/model_size.png` (gitignored).
