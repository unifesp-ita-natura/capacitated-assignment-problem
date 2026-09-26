# lightgbm_tuning

**Status:** confirmed — the pooled model beats `naive:mean` for the first
time, and most of the gain comes from one thing that was never a
hyperparameter question.

Stage 4 of `docs/plans/forecasting-feature-roadmap.md`.

## Goal

How much of the pooled model's remaining gap to `naive:mean` is
hyperparameters rather than features? And does the stage 1 explanation hold
— that a 7-leaf model was capacity-starved, so extra correlated columns
supplied split points rather than information?

## Hypothesis (written before running)

A small gain, 1 to 3 percent. The configuration in use until now (100 rounds
at lr 0.05, no early stopping, no subsampling, no penalty) was never tuned
against anything, but the sensitivity grid run before the roadmap showed a
monotone preference for smaller, shorter models — evidence there is little
structure to fit.

## Design note

A grid over `learning_rate`, `num_leaves`, `feature_fraction`,
`bagging_fraction` and `lambda_l2`, on top of everything stages 1–3
approved. Boosting runs up to 2000 rounds with early stopping on the last
cycle of each fold's **training** window, so the number of rounds is chosen
without seeing the cycle being forecast.

**Known limitation, and it is the serious one:** the grid is scored on the
same folds earlier stages used, so the winner's MAE is optimistic as an
estimate of unseen-data error. Stage 0 is what fixes this.

## Result (2026-09-22) — the benchmark falls

| configuration | MAE | vs naive:mean (1840.0) |
|---|---|---|
| stage 3 (100 rounds, no early stopping) | 1859.5 | +19.5 |
| **+ early stopping alone**, same lr and no subsampling | **1812.6** | **−27.4** |
| + lr 0.02 and `feature_fraction` 0.7 (grid winner) | **1803.9** | **−36.1** |

**Early stopping is almost the whole story.** It is worth **46.9 items** of
the 55.6-item gain; every other knob in the grid together adds 8.7. The old
configuration was not mis-tuned so much as *stopped far too early*: 100
rounds at lr 0.05 never finished learning, and nobody was measuring it
because there was no validation set to measure against.

That reframes the hypothesis. It predicted a 1–3% gain from tuning and got
3.0% — but from a knob that is not really a hyperparameter choice at all.

### It survives the seed check

`feature_fraction` below 1.0 finally makes `random_state` live, so the
roadmap's 5-seed criterion applies for the first time:

| seed | 0 | 1 | 2 | 3 | 4 | mean | sd |
|---|---|---|---|---|---|---|---|
| MAE | 1803.9 | 1806.9 | 1807.2 | 1807.5 | 1808.6 | **1806.8** | 1.8 |

**Every seed beats `naive:mean` (1840.0)**, the worst by 31.4 items. The
result is not a lucky draw.

### The grid had a flaw, and it was silent

`bagging_fraction` produced **identical** numbers at 0.7 and 1.0 — 48 rows,
24 distinct results. LightGBM ignores `bagging_fraction` unless
`bagging_freq > 0`, and `bagging_freq` stayed at its default of 0, so that
axis never ran. Re-tested properly on the winner:

| bagging_fraction (with `bagging_freq: 1`) | MAE |
|---|---|
| 0.7 | 1808.3 |
| 0.85 | 1809.4 |
| off | **1803.9** |

Bagging does not help here, so the conclusion is unchanged — but the grid as
published tested four axes, not five, and the duplicate rows are the only
reason it was caught.

### Axis effects, averaged over the grid

| axis | values |
|---|---|
| `num_leaves` | 7 → 1810.0 · 15 → 1817.3 · 31 → 1828.0 |
| `feature_fraction` | 0.7 → 1817.6 · 1.0 → 1819.2 |
| `learning_rate` | 0.02 → 1817.6 · 0.05 → 1819.3 |
| `lambda_l2` | 0 → 1817.9 · 5 → 1819.0 |

`num_leaves` is the only axis that moves the number meaningfully, and it
still prefers the smallest value offered — the same monotone preference for
small models the pre-roadmap grid found. Everything else is inside 2 items.

### The `num_leaves` curve, measured properly

The grid offered only 7, 15 and 31, so its preference for the smallest
value could not tell whether 7 was a peak or just the left edge of the
range. Swept on the winning configuration, 5 seeds per point:

| `num_leaves` | mean MAE | sd | range | vs naive:mean (1840.0) |
|---|---|---|---|---|
| 2 | 1814.0 | 1.7 | 1811.2–1815.6 | −26.0 |
| 3 | 1808.2 | 1.3 | 1807.0–1809.7 | −31.8 |
| 4 | 1807.9 | 2.0 | 1805.1–1810.4 | −32.1 |
| 5 | 1808.4 | 2.2 | 1805.4–1810.7 | −31.6 |
| **7** | **1806.8** | 1.8 | 1803.9–1808.6 | **−33.1** |
| 10 | 1809.9 | 1.5 | 1807.9–1811.9 | −30.1 |
| 15 | 1812.0 | 3.7 | 1807.2–1816.7 | −28.0 |
| 20 | 1815.5 | 3.6 | 1811.7–1820.5 | −24.4 |
| 31 | 1831.8 | 2.7 | 1827.7–1834.7 | −8.2 |
| 63 | 1834.2 | 3.4 | 1828.3–1836.4 | −5.8 |

**7 is an interior optimum, not an edge effect** — but a shallow one. The
span from 3 to 7 is 1.6 items against seed sds of 1.3–2.2, so those five
settings are a plateau, not a ranking; only 2 separates from it, by 7.2
items. Below the plateau the model runs out of places to put a pattern;
above 10 it degrades monotonically, and by 31 it has thrown away three
quarters of its margin over `naive:mean`.

The seed sd tracks the damage: 1.3–2.2 through the plateau, 3.4–3.7 at 15,
20 and 63. With ~633 sectors per cycle, 31 leaves means roughly 20
observations per leaf, so each leaf's estimate is mostly noise and which
noise depends on the seed. That is the same scarcity finding every other
stage produced, except self-inflicted: splitting the data finely enough
makes any regime data-poor.

Practical consequence: `num_leaves` is the one hyperparameter here worth
guarding. The whole roadmap gained 36.1 items over the benchmark; moving
this single knob from 7 to 31 gives back 25 of them.

### The stage 1 question, answered

Stage 1 found that order counts were worth 18.8 items and guessed the gain
might be capacity starvation rather than information. With real capacity
(5 seeds each):

| feature set | mean MAE | sd | range |
|---|---|---|---|
| with orders and volumes | **1806.8** | 1.8 | 1803.9–1808.6 |
| items only | 1814.9 | 2.5 | 1812.3–1817.9 |

**The guess was half right.** The order counts still carry a real
contribution — 8.1 items, with non-overlapping seed ranges — but that is
less than half of the 18.8 they were worth when the model was starved. So
roughly half of stage 1's headline gain was capacity, and half was
information.

### Decision

The tuned configuration is the new baseline. And the roadmap's own framing
needs a correction: it placed tuning fourth on the grounds that "tuning
before fixing the target is tuning the wrong model". That was right about
the order but wrong about the size — the single biggest gain in the whole
roadmap was sitting in an unmeasured stopping rule the entire time.


**Correction from a later experiment.** The claim above that early stopping
is "not really a hyperparameter choice at all" does not survive contact
with the two-year panel. `tuned_two_year` ran this same winner there and it
**lost** to the untuned configuration by 38.1 items, with early stopping
responsible for 29.1 of the damage: the stopping rule chose roughly 1100
rounds where fixed-budget runs show 100 is optimal on that panel. Early
stopping is a genuine, panel-dependent choice, and this configuration is a
one-year artifact — on a 12-cycle panel 100 rounds really did underfit, so
anything that let boosting run longer looked like a discovery.

## How to run

```bash
uv run python -m experiments.lightgbm_tuning.run
```

Writes `outputs/grid.csv` (every configuration, ranked).
