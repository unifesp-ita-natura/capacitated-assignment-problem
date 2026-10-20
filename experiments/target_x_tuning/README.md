# target_x_tuning

**Status:** confirmed — the hypothesis holds, and it reframes stage 3. The
ratio target's advantage more than triples once the model is tuned, and
tuning without it is not enough to beat the benchmark.

Crosses stage 3 with stage 4 of
`docs/plans/forecasting-feature-roadmap.md`.

## Goal

Stage 4's grid ran with `target: ratio` in its `fixed` block, so the two
stages were stacked but never **crossed**. Stage 3 measured the ratio
target at 17.4 items against an **untuned** model, and nobody checked
whether that value survives tuning — a model allowed to train far longer
could plausibly learn the 633 sector levels well enough not to need the
target's help. Does the ratio target still earn its place once the model is
tuned?

## Hypothesis (written before running)

The ratio target matters **more** after tuning, not less. Its job is to
stop the model spending capacity relearning 633 sector levels, and tuning
raised the round budget from 100 to whatever early stopping picks — a
longer budget is exactly what lets an untamed level target memorise those
levels instead of learning a pattern. If that is right, the tuned level arm
should lose to the tuned ratio arm by more than the 0.4 items that
separated them untuned.

## Design note

Two factors crossed on the one-year panel: target (level vs ratio) and
configuration (untuned stage-2/3 settings vs the stage-4 winner). The
untuned rows reproduce stages 2 and 3 exactly, so they are the control.
Tuned rows run 5 seeds each because `feature_fraction` 0.7 makes
`random_state` live. `min_child_samples` is 5 throughout — it is the schema
default, which is why stage 4's `fixed` block restating it changes nothing.

**Known limitation, inherited:** scored on the folds earlier stages
selected on. All 13 arms score all 3,689 points, so `mae_own` and
`mae_common` coincide and no arm is helped by skipping a hard fold.

## Result (2026-09-22) — hypothesis upheld, and then some

| | level target | ratio target | ratio's advantage |
|---|---|---|---|
| **untuned** | 1876.95 | 1859.55 | **17.4** |
| **tuned** (5 seeds) | 1862.39 ± 3.52 | **1806.84 ± 1.75** | **55.6** |
| tuning's gain | 14.6 | **52.7** | |

`naive:mean` = 1839.98 on the same 3,689 points.

**The two changes interact, and the interaction is larger than either main
effect measured alone.** Tuning is worth 14.6 items on a level target and
52.7 on a ratio target — **38.1 items of the gain exist only when both are
present**. Stage 4's headline number is not a property of the tuning; it is
a property of the pair.

**Tuning alone does not beat the benchmark.** The tuned level arm scores
1862.4, which is 22.4 items *worse* than `naive:mean`, and every one of its
5 seeds loses. Had stage 4's grid been run on the level target, the whole
stage would have read as a failure.

**The level target is also twice as seed-sensitive**: sd 3.52 against 1.75.
That is the signature of the mechanism — a model memorising 633 levels
depends on which columns each tree happened to draw.

### The control reproduces stage 3 exactly

The two untuned arms score 1876.95 and 1859.55 against stage 3's published
1877.0 and 1859.5, so the 17.4-item figure reproduces and the tuned arms
sit on the same panel, folds and basis as it. Nothing here rests on a
re-derived baseline.

### Decision

The ratio target is not an optional convenience adopted for fold coverage —
it is load-bearing, and it is what makes tuning pay. Any future experiment
that varies the boosting budget must hold the ratio target fixed, or it
will measure the wrong thing.

This is the second interaction the roadmap has found in one day, pointing
the same way: `tuned_two_year` showed the stage-4 configuration reverses
sign on a different panel, and this shows it reverses sign on a different
target. **Stage results in this plan are not additive and should stop being
reported as if they were.**

## How to run

```bash
uv run python -m experiments.target_x_tuning.run
```
