# experiments

One subdirectory per experiment (script + config), following the pattern:
a driver script that loads a `configs/experiments/<name>/*.yaml` and calls
into `src/`. Keep experiment-specific glue here; keep reusable logic in
`src/`.

Run artifacts belong under `<experiment>/outputs/` or `<experiment>/results/`
— both are gitignored — not committed alongside the driver script.

## Every experiment gets a README with a goal, hypothesis, and design note

Before writing the driver script, write `<experiment>/README.md`. This is
what makes an experiment folder findable and its result trustworthy months
later, instead of just "a script that ran once." Cover:

- **Name** — the experiment's directory name (so it's unambiguous which
  script/config/output set this note refers to).
- **Goal** — the question this experiment answers, in one sentence. Not
  "compare solvers" — "does HiGHS reach the same objective as Gurobi in
  under 2x the time on our largest instances?"
- **Hypothesis** — what you expect to find and why, stated before you run
  it. Getting this wrong is fine and informative; not writing it down
  before running removes the ability to tell whether a result was
  surprising.
- **Design note** — what's actually being varied, what's held fixed, what
  instances/config it runs against, and any known limitation of the setup
  (small sample, one seed, a shortcut that trades rigor for speed). This is
  what lets someone else judge how much to trust the result.
- **Status** — one of: `exploratory` (scoping, no result yet), `in-progress`
  (harness exists, full run pending), `confirmed` (result stands,
  reproducible from the cited config/code), `superseded` (a later
  experiment overtook this one — say which).

Update the status and add a **Result** section once the experiment has run;
don't leave the note as pre-registration only.

The same four things (goal, hypothesis, design note, status) also belong in
a `description:` block in the experiment's YAML config under
`configs/experiments/<name>/` — see `configs/experiments/README.md`. Keep
that version short; this README is where it's fine to go long, add
diagrams, and hold the **Result** section.

## Index

One line per experiment, updated as status changes:

| Experiment | Status | Goal |
|---|---|---|
| [compare_modeling](compare_modeling/README.md) | reference | Reference pattern for comparing solvers/formulations — not itself a research question, see its README. |
| [forecast_baseline](forecast_baseline/README.md) | exploratory | Run the naive candidate end-to-end through the shared forecasting harness against the real demand base. |
| [compare_forecasters](compare_forecasters/README.md) | confirmed | Rank naive, ARIMA and pooled LightGBM against each other on the real demand base through one shared harness. |
| [panel_order_counts](panel_order_counts/README.md) | confirmed | Does feeding the pooled LightGBM the order/volume counts of earlier cycles — series the panel used to discard — beat the items-only baseline? |
| [lightgbm_short_window](lightgbm_short_window/README.md) | confirmed | Does shortening the rolling-mean windows recover the fold the pooled model cannot predict, without costing accuracy? |
| [relative_target](relative_target/README.md) | confirmed | Does fitting items divided by the sector's running mean beat fitting the level itself? |
| [cycle_factor](cycle_factor/README.md) | confirmed | Does a factor shared by every sector in a cycle carry usable signal? (No — it has no memory.) |
| [lightgbm_tuning](lightgbm_tuning/README.md) | confirmed | How much of the gap to naive:mean is hyperparameters? (Mostly early stopping — and it beats the benchmark.) |
| [two_year_panel](two_year_panel/README.md) | confirmed | With a second year of history, does the pooled model beat naive:mean, and does the year-ago cycle help? (Yes to both.) |
| [tuned_two_year](tuned_two_year/README.md) | confirmed | Do the stage 4 tuning and the stage 6 two-year panel compose? (No — tuning costs 38 items there; early stopping overshoots the round budget tenfold.) |
| [target_x_tuning](target_x_tuning/README.md) | confirmed | Do the stage 3 ratio target and the stage 4 tuning compose? (They interact — 38 of the 53 items exist only when both are present, and tuning alone loses to naive:mean.) |
| [daily_output](daily_output/README.md) | confirmed | Score forecasts per CD-day instead of per cycle: does the candidate ranking hold, and does a per-sector curve beat a flat spread? (Ranking holds; the curve cuts CD-day error 5.4%.) |
