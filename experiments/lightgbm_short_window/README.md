# lightgbm_short_window

**Status:** confirmed — coverage fully recovered and the common-subset error
improved, but for a narrower reason than the hypothesis claimed.

Stage 2 of `docs/plans/forecasting-feature-roadmap.md`.

## Goal

Does shortening the rolling-mean windows from `[3, 6]` to `[2, 3]` recover
the fold the pooled LightGBM cannot predict at all, without costing accuracy
on the folds it already predicts?

## Hypothesis (written before running)

It recovers coverage and does not hurt. A 6-cycle rolling mean needs 6 prior
complete cycles, so the first fold's training table comes out **empty** and
606 target points are forfeited. Shortening the window moves the binding
constraint to `lag_4`, which the first fold can satisfy. Accuracy was
expected to be roughly flat, because stage 1 suggested the model is
capacity-limited rather than information-limited.

## Design note

**What varies:** `rolling_windows`, `[3, 6]` → `[2, 3]`.

**What is held fixed:** everything stage 1 approved, including
`companion_lags: [1, 2, 3, 4]`, plus the split, the target, the objective
and every hyperparameter.

**Coverage changes on purpose.** Unlike stage 1, the two LightGBM rows do
*not* score the same points — that is the thing being tested.
`src.forecasting.comparison.compare` ranks them on the intersection, which
is the honest accuracy comparison, while `n_missing` carries the coverage
result.

**Naming.** The two rows differ only in a field the generated candidate name
doesn't carry, so `LightGBMParams.label` (added for this experiment) names
them explicitly.

**Known limitation:** same folds earlier stages selected on, so the numbers
stay optimistic until stage 0's holdout exists.

## Result (2026-09-22) — hypothesis upheld, mechanism narrower than claimed

Errors in items, no thousands separator.

| candidate | MAE (common subset) | MAE (own) | scored | **skipped** |
|---|---|---|---|---|
| naive:mean | **1834.3** | 1840.0 | 3689 | 0 |
| lightgbm+orders w[2,3] | 1862.5 | 1877.0 | **3689** | **0** |
| lightgbm+orders w[3,6] | 1873.2 | 1873.2 | 3083 | 606 |

**Coverage is fully recovered.** The short-window model forfeits nothing: it
predicts all 3689 points, the same as the naive benchmark. That removes the
pooled model's standing disadvantage — until now it was the only candidate
family that simply had no answer for an entire fold.

**And the common-subset error improves** by 10.7 items (0.57%), so the
recovery did not come at the cost of accuracy on the shared points.

### Where the gain actually comes from

| fold (origin) | naive:mean | w[3,6] | w[2,3] | w[2,3] − w[3,6] |
|---|---|---|---|---|
| 202606 | 1842.0 | *no prediction* | 1926.1 | recovered |
| 202607 | 1690.8 | 1915.4 | **1739.1** | **−176.3** |
| 202608 | 1672.6 | 1700.4 | 1741.1 | +40.7 |
| 202609 | 2309.8 | 2231.2 | 2265.5 | +34.3 |
| 202610 | 1829.6 | 1795.8 | 1844.1 | +48.3 |
| 202611 | 1656.1 | 1687.6 | 1683.0 | −4.6 |

The whole common-subset gain is **one fold**. In 202607 — the earliest fold
the long-window model could predict at all, where it trained on 571 rows —
shortening the window is worth 176 items. In the three middle folds, where
training data is plentiful, the **6-cycle window is actually better**, by
about 41 items on average.

So the honest mechanism is not "shorter windows are a better feature set".
It is: **a 6-cycle mean is unaffordable early and slightly better late.**
What the change really buys is that the model stops being data-starved in
the folds where history is thin.

### Decision

`w[2,3]` is carried into stage 3. Two reasons, both worth stating:

1. Full coverage is not a tie-breaker, it is a requirement. A sector with no
   forecast is a sector the assignment model cannot place.
2. It is also better on the shared points, so nothing is traded away.

**But this is provisional, and stage 6 should revisit it.** Once the second
year of history is in the panel (`Unifesp_Demanda_v2`, verified usable in
`docs/context/natura-client-answers.md`), every fold becomes data-rich — the
regime where `[3, 6]` won here. The longer window may well come back, and
the year-ago window becomes affordable for the first time.

**Still behind the benchmark.** `naive:mean` leads by 28.2 items on the
common subset (1.5%) and by 37.0 on full coverage (2.0%). The recovered fold
202606 is itself a loss against naive (1926.1 vs 1842.0).

## How to run

```bash
uv run python -m experiments.lightgbm_short_window.run
```
