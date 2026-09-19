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
