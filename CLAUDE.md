# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This problem is a Generalized Assignment Problem (GAP) where we have a set of sectors and their respective historical orders. Each sector is grouped in a specific block and sub-block which determines the sector's start date for each cycle. The goal is to assign each sector to a specific block and sub-block in order to provide a more balanced order distribution across the blocks and sub-blocks. The solver will determine the optimal assignment of sectors to blocks and sub-blocks, taking into account the historical orders and the start dates for each cycle. The output of a solve will be a mapping of sectors to their assigned blocks and sub-blocks, along with any relevant metrics or statistics about the assignment.

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

- `src/solver/README.md` — solve loop and model definition
- `src/config/README.md` — config loading and schema
- `src/persistence/README.md` — how results are written and read back

Cross-cutting references:

- `docs/context/natura-client-answers.md` — o que o cliente (Natura) já
  respondeu sobre regras de negócio, capacidade dos CDs e os dados, mais as
  dúvidas ainda em aberto. Consulte antes de assumir qualquer premissa de
  negócio.
- `docs/plans/forecasting-feature-roadmap.md` — plano incremental de testes de
  novas features para a previsão de demanda, com o protocolo de avaliação que
  toda etapa precisa respeitar.
- `docs/architecture/` — put runtime-output layout and repository conventions here as the project grows
- `docs/conventions/` — commit format, module headers, function design, testing

## Environment

The MIP solver defaults to Gurobi (commercial, requires a license — see
`src/solver/README.md`). Natura, the client, already has a Gurobi license,
so this is fine for the proposed solution. If no working Gurobi
license/install is found on the machine running it, the code falls back to
HiGHS automatically (open-source, bundled via `highspy`) and logs a warning,
so the project stays runnable without a license too.

## Conventions

These apply whenever you touch code, commit, or write a paper. Each is short — read the one relevant to what you're doing:

- `docs/conventions/commits.md` — commit message format, types, staging rules
- `docs/conventions/module-headers.md` — the one-sentence module docstring convention
- `docs/conventions/function-design.md` — small, single-purpose function guidelines
- `docs/conventions/testing.md` — mocking the solver, coverage targets, ruff/radon commands

**Before writing a new helper in `experiments/`, check if `src/` already has one.**
Experiment scripts are orchestration/glue around real logic; if you're writing
config loading, data loading/validation, metric computation, plotting, or
file/manifest I/O, search `src/` first (`grep -r` for the operation, or check
the relevant `src/*/README.md`). Only write it in `experiments/` if it's
genuinely specific to that one experiment. If two or more experiment scripts
end up with similar helper code, that's a signal to move it to `src/`
immediately rather than letting a third copy appear.

## Formal proofs (optional)

`formal/` has a Lean 4 / Mathlib scaffold for formalizing paper theorems. See `formal/README.md`. 

## AGENTS.md

`AGENTS.md` holds agent-specific behavioral defaults (what to check before committing, how to handle ambiguity). Read it alongside this file.
