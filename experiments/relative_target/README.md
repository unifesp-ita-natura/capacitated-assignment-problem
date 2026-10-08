# relative_target

**Status:** confirmed — the ratio target wins, and closes most of the
remaining gap, but `naive:mean` still leads.

Stage 3 of `docs/plans/forecasting-feature-roadmap.md`.

## Goal

Does training the pooled LightGBM on **items divided by the sector's own
running mean**, instead of on the item count itself, beat training on the
level?

## Hypothesis (written before running)

It wins, and it is the most structural change in the roadmap. On the level
target the model must learn 633 sector sizes through a 633-value
categorical using `num_leaves: 7` — six splits in total, which cannot
separate them. Dividing by the running mean removes that job: the model
predicts a *correction* on top of the naive:mean forecast, so a model that
learns nothing reproduces the benchmark instead of falling short of it.

## Design note

**What varies:** `LightGBMParams.target`, `"level"` → `"ratio"`.

**The denominator** is the sector's expanding mean over its earlier cycles
(`features.LEVEL_REFERENCE`) — precisely what `naive:mean` predicts for that
row. This is a deliberate deviation from the roadmap, which said
`rolling_mean_k`: the expanding mean makes the comparison interpretable,
because a fitted ratio of 1.0 everywhere reproduces the benchmark exactly.
It is never a feature, only the denominator, so the feature set is
unchanged and the target is the only thing that moves.

**Held fixed:** everything stages 1 and 2 approved (`companion_lags
[1,2,3,4]`, `rolling_windows [2,3]`), the split, and every hyperparameter.
All three rows have full coverage, so the comparison runs on all 3689
points.

**Known limitation:** the same folds earlier stages selected on.

## Result (2026-09-22) — hypothesis upheld, benchmark still ahead

| candidate | MAE | scored | skipped |
|---|---|---|---|
| naive:mean | **1840.0** | 3689 | 0 |
| ratio target (stage 3) | 1859.5 | 3689 | 0 |
| level target (stage 2) | 1877.0 | 3689 | 0 |

The ratio target is worth **17.4 items (0.93%)**. Cumulatively the pooled
model has gone from 57.7 items behind `naive:mean` at the start of the
roadmap to **19.6 behind** — the gap has shrunk by two thirds in three
stages.

### Fold by fold

| fold | naive:mean | level | ratio | gain | vs naive:mean |
|---|---|---|---|---|---|
| 202606 | 1842.0 | 1926.1 | 1973.3 | −47.2 | +131.3 |
| 202607 | 1690.8 | 1739.1 | 1744.1 | −4.9 | +53.3 |
| 202608 | 1672.6 | 1741.1 | 1679.3 | **+61.8** | +6.7 |
| 202609 | 2309.8 | 2265.5 | 2268.6 | −3.1 | **−41.2** |
| 202610 | 1829.6 | 1844.1 | 1794.0 | **+50.2** | **−35.6** |
| 202611 | 1656.1 | 1683.0 | 1674.3 | +8.7 | +18.2 |

Only **3 of 6** folds improve, but the two that do are large (+61.8, +50.2)
and the losses are small, except in 202606. That fold — the data-poorest,
recovered only in stage 2 — is where the ratio target hurts most (−47.2),
and it alone supplies most of the remaining deficit against `naive:mean`
(+131.3 out of a 19.6 average).

**The pattern from stage 2 repeats:** every change so far helps where data
is plentiful and hurts where it is thin. That is a consistent story about
this base, not a coincidence, and it is the strongest argument yet for
stage 6 — the only stage that adds history rather than rearranging it.

The model now beats `naive:mean` on **2 of 6 folds**, the same two as
before (202609, 202610).

### Decision

The ratio target is carried into stage 4. Tuning now runs against a target
the model can actually represent.

## How to run

```bash
uv run python -m experiments.relative_target.run
```
