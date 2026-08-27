# experiments-template

A bare scaffold for optimization-research group projects.

New to the team or new to software engineering in general? Start with
`CONTRIBUTING.md` instead of this file — it's a step-by-step, plain-English
walkthrough, with `docs/glossary.md` for any unfamiliar terms.

## Getting started

1. Rename the project in `pyproject.toml`, `configs/default.yaml`, and
   (if using `formal/`) `formal/lakefile.toml`.
2. Fill in the `TODO`s in `CLAUDE.md` with your actual problem statement.
3. Set real owners in `.github/CODEOWNERS` (currently `@TODO-set-owner`),
   then turn on branch protection requiring the `CI / test` check and a
   CODEOWNERS review.
4. Replace `src/solver/mip/example.py` with your real model, following the
   same Pyomo `ConcreteModel` → `SolverFactory(solver_name).solve(...)`
   pattern. Add other solving approaches (CP-SAT, heuristics) as sibling
   families under `src/solver/` — see `src/solver/README.md`.
5. Point `.env` (gitignored) at your Gurobi license — see
   `src/solver/README.md`.
6. If you don't need Lean formal proofs, delete `formal/` and the matching
   `AGENTS.md` bullet.

```bash
make setup            # one-time per clone — installs deps, hooks, runs tests
uv run python -m src.cli.main
```

## Pre-commit hook

`make install-hooks` (or `scripts/install-hooks.sh`) symlinks
`scripts/hooks/pre-commit` into `.git/hooks/pre-commit`. It runs on every
commit:

- Blocks any staged file over 5MB (override per-commit with
  `MAX_COMMIT_FILE_SIZE_MB=<n>`) — see `data/README.md` for why data
  doesn't belong in git
- `uv run nbstripout` on staged `.ipynb` files — strips cell outputs and
  re-stages them, so notebook outputs never land in git history
- `uv run ruff format --check` on staged Python files
- `uv run ruff check` on staged Python files
- `uv run radon cc -s -n B` on staged Python files — fails the commit if any
  function is below complexity grade A

It's a per-clone setup step (git doesn't version `.git/hooks/`), so run it
again after cloning or creating a new worktree.

## What's here

- `CONTRIBUTING.md` — plain-English, step-by-step workflow guide for anyone
  new to git/software engineering; `docs/glossary.md` backs it with term
  definitions
- `src/config/` — YAML config loading via a typed `RunConfig` (pydantic)
- `src/solver/` — solving approaches as sibling families, each reporting into
  the same `SolveResult` so they're directly comparable: `mip/` (Pyomo +
  Gurobi/HiGHS, with a working example), `cpsat/` and `heuristics/`
  (placeholders — see their READMEs). The "mock the solver in tests"
  convention is already wired up in `mip/`.
- `src/persistence/` — `SolveResult` schema + CSV writer, the common shape
  every solver/formulation comparison reports into
- `src/cli/` — a thin entrypoint tying config + solver together
- `docs/conventions/` — commit format, module headers, function design,
  testing/mocking conventions
- `docs/literature/` — one file per paper/source read for the project, plus
  an index; see `docs/literature/README.md`
- `formal/` — optional Lean 4 / Mathlib scaffold for formalizing paper
  theorems (delete if unused)
- `configs/` — `default.yaml` (project defaults) and `experiments/<name>/`
  (per-experiment YAML, each required to carry a goal/hypothesis/
  design-note/status `description:` block — see
  `configs/experiments/README.md`); `Makefile` clean targets grow alongside
  it
- `data/` — `instances/`, `processed/`, `generated/`; contents are
  gitignored (only the directory skeleton is tracked) — see `data/README.md`
  for why, and where raw data that can't be regenerated should actually live
- `notebooks/` — exploratory Jupyter notebooks (throwaway; promote real logic
  into `src/`)
- `experiments/` — one subdirectory per experiment (driver script + config +
  README with the same goal/hypothesis/design-note/status structure); run
  outputs are gitignored, not committed. `compare_modeling/` is a working
  reference: runs the example model across multiple solvers and writes a
  comparison table.
- `.github/workflows/ci.yml` — lint (`ruff`) + test (`pytest --cov=src`) on
  push/PR to `main`
- `.github/CODEOWNERS` — placeholder owners; fill in before enabling branch
  protection
- `.github/PULL_REQUEST_TEMPLATE.md` — fill-in-the-blanks PR description
  checklist

## Repository settings

Branch protection itself is a repo-settings toggle done in GitHub — turn it on once CODEOWNERS has real owners. 
