# configs/experiments

One subdirectory per experiment, holding its YAML config(s) (e.g.
`smoke.yaml` for a fast local check, `full.yaml` for the real run). Loaded
by that experiment's driver script in `experiments/<name>/`.

## Every experiment config carries its own description

Put a `description:` block at the top of each experiment config, alongside
`experiment_name:`. This is the same goal/hypothesis/design-note/status
convention described in `experiments/README.md` — living in the config
itself means it's the one thing guaranteed to travel with every run: if a
results file or log gets copied elsewhere, or an experiment's driver script
changes six months from now, the config that actually produced a given
result still says what question it was answering and why.

```yaml
experiment_name: compare_modeling_gurobi_vs_highs
description:
  goal: >
    One sentence: the question this run answers.
  hypothesis: >
    What you expect to find and why, written before running it.
  design_note: >
    What's varied, what's held fixed, what instances/config this runs
    against, and any known limitation (small sample, one seed, a shortcut
    that trades rigor for speed).
  status: exploratory  # exploratory | in-progress | confirmed | superseded | reference

# ... experiment-specific fields below ...
```

`status` uses the same vocabulary as `experiments/README.md` — update it in
place as the experiment progresses, don't leave it at `exploratory` once
you have a result.

Nothing currently parses `description` into a typed schema (no experiment
here loads YAML config yet — `compare_modeling` hardcodes its solver list
directly in `run.py`). Add a validated `description` field to your config
schema in `src/config/schema.py` once a real experiment needs to read one;
until then this is a documented convention, not an enforced one.
