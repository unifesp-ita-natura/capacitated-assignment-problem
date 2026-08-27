# configs

- `default.yaml` — project-wide defaults (`RunConfig`, `src/config/schema.py`).
- `experiments/<name>/*.yaml` — per-experiment configs; see
  `experiments/README.md` for the required
  goal/hypothesis/design-note/status structure.
- `cluster/` — cluster job submission configs (PBS/Slurm scripts etc.), if
  this project runs experiments on a shared cluster.
