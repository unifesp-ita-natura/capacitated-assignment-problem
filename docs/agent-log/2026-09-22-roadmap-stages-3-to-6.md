# Roadmap stages 3 to 6: ratio target, tuning, cycle factor, two-year panel

**Date:** 2026-09-22

## Task

Run stages 3 through 6 of `docs/plans/forecasting-feature-roadmap.md` in
order, each pre-registered and recorded before moving to the next.

## Outcome

**The pooled model beats `naive:mean`, on both panels.** On the one-year
panel it went from 19.5 items behind to 36.1 ahead (1803.9 against 1840.0),
in all five seeds. On the two-year panel it wins by 306.5 items, 11.0%.

Stage by stage:

- **Stage 3 (ratio target) won** — 1877.0 → 1859.5, the largest gain until
  stage 4.
- **Stage 4 (tuning) broke the benchmark**, but almost entirely through
  **early stopping**: worth 46.9 of the 55.6-item gain, with every other knob
  in the grid adding 8.7 between them. The old configuration was not
  mis-tuned, it was stopping far too early, and nothing was measuring that
  because there was no validation set.
- **Stage 5 (shared cycle factor) was rejected outright** — the most
  informative failure: the ±20% swing that hits every sector at once is real,
  but its lag-1 autocorrelation is **+0.110**, so it cannot be forecast from
  its own past.
- **Stage 6 (two-year panel) is the biggest win**, and the year-ago lag is
  worth 64.1 items on its own — the first feature in the whole roadmap that
  carries information the panel did not already contain.

## What changed

- `src/forecasting/features.py` — `LEVEL_REFERENCE` (each row's running mean
  of its sector's earlier cycles) and `complete_feature_frame`, which
  replaces `build_training_table` as the single primitive. A caller now cuts
  the target, the validation split and the ratio denominator from the same
  rows the features came from.
- `src/forecasting/candidates/lightgbm.py` — the ratio target (stage 3) and
  early stopping on the training window's last cycles (stage 4).
- `src/forecasting/candidates/cycle_factor.py` (new) — `level × factor`,
  registered as its own candidate family.
- `src/config/schema.py` — `target`, `feature_fraction`, `bagging_fraction`,
  `bagging_freq`, `lambda_l2`, `early_stopping_rounds`, `validation_cycles`,
  and `CycleFactorParams`.
- `experiments/{relative_target,lightgbm_tuning,cycle_factor,two_year_panel}/`
  and their configs (new).
- `experiments/two_year_panel/prepare.py` — builds a two-year demand CSV from
  the client's v2 workbook joined to the 2024–2026 cycle calendar, via
  `python-calamine` (already a dependency; it reads the 400k-row sheet in
  seconds where openpyxl takes minutes).

## Notes

**A repeated pattern, now the main finding of the whole roadmap.** Every
change that helped — stage 1, stage 2, stage 3 — helped in the folds with
plentiful training data and *hurt* in the data-starved ones. In stage 3 the
worst fold (202606, the poorest) alone supplies most of the remaining deficit
against the benchmark. That is one consistent story about this base rather
than three coincidences, and it is the strongest argument for stage 6, the
only stage that adds history instead of rearranging it.

**An estimator bug worth remembering.** The cycle factor was first estimated
as the mean of `items / running_mean` across sectors. The series opened at
**3.51** and decayed toward 1 — it looked like a spectacular demand surge and
was an artifact: early rows divide by a mean built from one or two cycles,
and the mean of ratios is not the ratio of means. Switching to a ratio of
sums cut the dispersion fivefold (σ 0.701 → 0.138). The candidate still lost,
but the first number is exactly the kind of result that would have been
reported as a discovery.

**A correction to the roadmap's own reasoning.** Stage 5 was justified by the
*total items per cycle* series (0.62 … 1.20 indexed). That quantity mixes the
cycle effect with changes in which sectors were observed. Measured against
what each sector's own history expected, the effect is smaller and has
essentially no memory. The roadmap text has been corrected in place.

**Stage 4's seed question, revisited.** Stage 1 skipped multiple seeds
because the configuration had no stochastic component. `feature_fraction`
below 1.0 is what finally made `random_state` live, and the five-seed check
holds: mean 1806.8, sd 1.8, and every seed beats the benchmark.

**A silent flaw in the tuning grid, caught only by duplicate rows.**
`bagging_fraction` produced identical numbers at 0.7 and 1.0 — 48
configurations, 24 distinct results. LightGBM ignores `bagging_fraction`
unless `bagging_freq > 0`, which stayed at its default of 0, so that axis
never ran. Re-tested properly, bagging does not help, so the conclusion
stands — but the published grid tested four axes, not five, and the only
reason it was noticed is that the duplicates were exact.

**Stage 1's open question, answered.** With real capacity the order counts
are still worth 8.1 items (seed ranges non-overlapping), against the 18.8
they were worth when the model was starved. So roughly half of stage 1's
headline gain was capacity and half was information — the guess was half
right.

**What stage 6's headline needs attached to it.** Two thirds of the 306-item
win over `naive:mean` is the model; the other third is recency, since a mean
over 20+ cycles spans a level shift. A mean of the last 6 cycles alone
scores 2677.6 against `naive:mean`'s 2775.4. The 11% figure should never be
quoted without that decomposition. The panel is also different (728 sectors
× 32 cycles), so its error levels are not comparable to any earlier stage.
