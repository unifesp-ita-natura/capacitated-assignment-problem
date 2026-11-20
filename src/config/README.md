# `src/config`

Loads and validates the run configuration.

- `schema.py` — `RunConfig`, a pydantic model describing project name,
  modeling layer, solver choice, on-disk paths, forecast model and the
  end-to-end `pipeline` section (`PipelineConfig`, used by `src.main`).
- `loader.py` — `load_config(path)` reads a YAML file into a `RunConfig`.

The application defaults to `forecast.params.model: hierarchical` with
`target_level: sector` (region → management → sector). The level output is one
row per scenario-sector-cycle and the daily output is one row per sector-day.
Legacy city output requires `target_level: city`, with `city_grouping: sector`
or `management`. All modes feed the daily level × shape flow; see
[the combined forecast guide](../../docs/guides/combined-forecast.md).

Extend `RunConfig` as the project grows (experiment parameters, generator
settings, solver tuning). Keep new fields typed and validated rather than
reading raw dicts downstream.
