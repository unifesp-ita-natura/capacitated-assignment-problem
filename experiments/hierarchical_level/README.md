# Hierarchical level experiments

Compare moving-average baselines, ETS/LightGBM distribution and nonnegative
weighted least-squares reconciliation at configurable organization levels.
This runner does not change the existing level or shape executors.

The standard comparison emits exactly `direct_ma_5`,
`distribution_ets_lgbm` and `reconciliation_ets_lgbm`, with 20 closed training
cycles in the supplied standard configurations. The moving average is never
distributed or reconciled. Archived configurations under
`configs/experiments/hierarchical_level/historical/` explicitly opt into
`comparison: historical` to reproduce the previous full candidate set.

See [the Portuguese guide](../../docs/guides/hierarchical-level.md) for setup,
YAML settings, feature definitions, time cutoffs, scenario inputs and results.

```powershell
.\.venv\Scripts\python.exe -m experiments.hierarchical_level.run configs/experiments/hierarchical_level/full.yaml
```

Choose a new `paths.run_dir` before rerunning. Execution artifacts are ignored
by Git; source, configurations and documentation remain versionable.

For controlled leave-one-feature-block-out analysis with a reproduced full
control, use the [feature ablation guide](../../docs/guides/hierarchical-feature-ablation.md).
The separate `experiments.hierarchical_level.ablation` runner reuses fold
feature tables and archives each variant without changing production defaults.
