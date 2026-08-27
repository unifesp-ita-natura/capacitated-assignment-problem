# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TODO: replace this section with your project's actual problem statement — what
is being modeled, what the solver decides, and what a "solve" produces. Keep
it to a short paragraph; put details in `src/*/README.md` files, not here.

This scaffold ships with a minimal Pyomo + Gurobi wiring
(`src/solver/README.md`) and a config-loading pattern
(`src/config/README.md`) as a starting point — replace or extend them with
your actual model.

## Commands

```bash
# Install dependencies (Python 3.13+ required)
uv sync

# One-time per-clone setup: installs the pre-commit hook (ruff + radon gate)
make install-hooks

# Run tests
uv run pytest -q

# Run a single test file
uv run pytest tests/solver/mip/test_example.py

# Format and lint Python files
uv run ruff format src tests experiments
uv run ruff check src tests experiments

# Demo solve (default config)
uv run python -m src.cli.main
```

## Architecture

TODO: as the codebase grows, document the pipeline here the way the original
template did — one or two sentences per package, linking to that package's
own `README.md` for detail. Example:

- `src/solver/README.md` — solve loop and model definition
- `src/config/README.md` — config loading and schema
- `src/persistence/README.md` — how results are written and read back

Cross-cutting references:

- `docs/architecture/` — put runtime-output layout and repository conventions here as the project grows
- `docs/conventions/` — commit format, module headers, function design, testing

## Environment

TODO: document your solver license setup here (this scaffold assumes a
Gurobi license via `.env` — see `src/solver/README.md`). **Do not call the
real solver directly in tests** — see `docs/conventions/testing.md`.

## Conventions

These apply whenever you touch code, commit, or write a paper. Each is short — read the one relevant to what you're doing:

- `docs/conventions/commits.md` — commit message format, types, staging rules
- `docs/conventions/module-headers.md` — the one-sentence module docstring convention
- `docs/conventions/function-design.md` — small, single-purpose function guidelines
- `docs/conventions/testing.md` — mocking the solver, coverage targets, ruff/radon commands

## Formal proofs (optional)

`formal/` has a Lean 4 / Mathlib scaffold for formalizing paper theorems, if
your project needs one. See `formal/README.md`. If you don't need formal
proofs, delete `formal/` and drop the requirement from `AGENTS.md`.

## AGENTS.md

`AGENTS.md` holds agent-specific behavioral defaults (what to check before committing, how to handle ambiguity). Read it alongside this file.
