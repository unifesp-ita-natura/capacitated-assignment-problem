# tuned_two_year

**Status:** confirmed — the hypothesis failed. Stages 4 and 6 do not
compose; the stage 4 configuration is *worse* than the untuned one on the
two-year panel, and the knob responsible is the one that carried stage 4.

Crosses stage 4 with stage 6 of `docs/plans/forecasting-feature-roadmap.md`.

## Goal

Stage 6 ran the **untuned** stage-3 configuration on the two-year panel, so
the tuning that broke the benchmark on the one-year panel had never met the
extra history. Do the two gains add up?

## Hypothesis (written before running)

They compose, but not fully. Stage 4's gain came almost entirely from early
stopping — the old 100-round budget was simply too short — and a panel with
32 cycles instead of 12 gives a pooled model more rows per fold, which is
exactly the regime where a longer boosting budget should pay off *more*,
not less. Against that, part of stage 6's margin over `naive:mean` is
recency rather than structure, and tuning cannot add to that part.

## Design note

Two factors crossed on the two-year panel: the configuration (untuned
stage-3 settings vs the stage-4 winner) and the year-ago features (lag 19
plus the annual rolling mean, present or absent). The untuned rows
reproduce stage 6 exactly, so its numbers are the control. The tuned rows
run 3 seeds each, because `feature_fraction` 0.7 makes `random_state` live;
3 rather than 5 because each fit on this panel is far more expensive than
on the one-year one.

**Known limitations, inherited:** this panel is 728 sectors × 32 cycles, so
its error levels compare to nothing outside this table; and everything here
is still scored on folds earlier stages selected on.

## Result (2026-09-22) — the hypothesis fails

| configuration | MAE | vs `naive:mean` (2708.6) |
|---|---|---|
| **untuned + year-ago (stage 6)** | **2408.5** | **−300.1** |
| tuned + year-ago (3 seeds) | 2446.5 ± 2.1 | −262.1 |
| untuned, no year-ago | 2476.2 | −232.4 |
| tuned, no year-ago (3 seeds) | 2500.9 ± 4.1 | −207.7 |

Tuning **costs** 38.1 items with the year-ago features and 24.7 without.
All three seeds of each tuned arm lose to their untuned control, the best
of them by 35.6 items, so this is not a seed draw.

The same configuration was worth **−36.1 items** on the one-year panel.
Moving it to a panel with nearly three times the cycles flips its sign.

### Which knob does the damage

One factor at a time, all with the year-ago features:

| configuration | MAE | effect of the knob added |
|---|---|---|
| A. untuned control | **2408.5** | — |
| B. + early stopping | 2437.6 | **+29.1 (worse)** |
| C. + learning rate 0.02 | 2453.6 | **+16.0 (worse)** |
| D. + `feature_fraction` 0.7 | 2447.5 | −6.1 (better) |

The pre-registered guess — that `feature_fraction` would dilute the lag-19
column and cause the damage — is **wrong**: column subsampling is the only
knob of the three that helps here. **Early stopping is the villain**, the
same knob that was worth 46.9 of stage 4's 55.6-item gain.

### Why early stopping hurts: the budget, not the sacrificed cycle

Two mechanisms could explain it, with opposite fixes. Early stopping holds
the last training cycle out of the fit set (`_split_for_early_stopping`),
so the model never trains on the freshest cycle — and stage 6 showed this
panel rewards recency. Alternatively the long budget itself overfits.

Fixed budgets at the untuned learning rate, no early stopping, so every row
trains on **all** cycles and only the number of rounds varies:

| rounds | MAE | vs 100 |
|---|---|---|
| **100** | **2408.5** | — |
| 200 | 2411.0 | +2.5 |
| 400 | 2424.0 | +15.5 |
| 800 | 2434.2 | +25.7 |
| 1600 | 2443.3 | +34.9 |

**The budget is the cause.** More rounds degrade accuracy monotonically
with the sacrificed cycle removed from the question entirely. And the
arithmetic closes: arm B scored 2437.6, which interpolates between the 800-
and 1600-round rows at roughly **1100 rounds**. Early stopping was not
paying a price for the held-out cycle — it was choosing a budget an order
of magnitude too long and the held-out cycle cost approximately nothing.

That leaves an open question worth one experiment: *why* does the stopping
rule overshoot so far? Two candidates. The validation set is a single cycle
(~728 rows) with 50 rounds of patience, which may simply be too noisy a
signal to stop on. Or the metric disagrees with the scoreboard — boosting
stops on L2 of the **ratio** target while the experiment ranks on MAE of
**items**, and on a heavy-tailed target those two can move in opposite
directions for a long stretch.

### Decision

**The two-year baseline stays untuned.** Stage 4's winner is a one-year
artifact: on a short panel 100 rounds genuinely underfit, so anything that
let boosting run longer looked like a discovery. With 32 cycles the same
budget overfits instead.

The wider lesson contradicts how stage 4 was written up. That README called
early stopping "not really a hyperparameter choice at all", on the grounds
that it just fixes a budget nobody had measured. This experiment shows it
*is* a choice, and a panel-dependent one: the rule picked ~1100 rounds
where 100 was optimal. Hyperparameters found on one panel do not transfer
to another panel of the same problem, and the roadmap should stop assuming
that a knob validated once stays validated.

## How to run

```bash
uv run python -m experiments.tuned_two_year.run
```

Requires `data/processed/demand_two_years.csv`, built by
`uv run python -m experiments.two_year_panel.prepare`.
