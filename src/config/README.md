# `src/config`

Loads and validates the run configuration.

- `schema.py` — `RunConfig`, a pydantic model describing project name,
  modeling layer, solver choice, and on-disk paths.
- `loader.py` — `load_config(path)` reads a YAML file into a `RunConfig`.

Extend `RunConfig` as the project grows (experiment parameters, generator
settings, solver tuning). Keep new fields typed and validated rather than
reading raw dicts downstream.
