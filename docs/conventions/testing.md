# Testing Conventions

Tests live in `tests/`. `conftest.py` provides shared fixtures.

## Solver

**Prefer solving for real with HiGHS over mocking.** HiGHS ships as a direct
dependency (`highspy`) and needs no license, so tests that exercise solving
should call `solve(..., solver="appsi_highs")` (or let a default that falls
back to HiGHS resolve on its own) and assert on the real `SolverResults`
(termination condition, objective) rather than on a `MagicMock`'s call log.

**Only mock to simulate "Gurobi has no license."** A dev machine with a
working Gurobi license can't produce that condition for real, so stub out
just `pyo.SolverFactory("gurobi")` (e.g. a small fake with
`available() -> False`) while leaving every other solver name — in
particular the HiGHS fallback — going through the real `pyo.SolverFactory`.
See `tests/solver/mip/test_example.py` and `tests/solver/mip/test_block_assignment.py`
for the pattern, and `src/solver/README.md` for the fallback behavior being
tested.

## Commands

```bash
uv run pytest -q
uv run pytest tests/solver/mip/test_example.py
uv run pytest --cov=src
```

Run after broad Python edits:

```bash
uv run ruff format src tests experiments
uv run ruff check src tests experiments
uv run radon cc -s -a src
```

Code must stay in complexity grade A per `radon cc`.

A pre-commit hook (`scripts/hooks/pre-commit`, installed once per clone via
`make install-hooks`) runs `ruff format --check`, `ruff check`, and a
`radon cc -n B` grade gate on staged Python files automatically — see
`AGENTS.md`.

## Coverage targets

Use risk-based coverage targets, not a single blanket number:

* 80%+ on pure logic modules.
* 90%+ on critical solver/core modules.
* Lower coverage is acceptable on thin CLI, plotting, or workflow glue code
  when behavior is already covered indirectly by focused tests elsewhere.

Add unit tests where `pytest --cov=src` shows a gap in the modules that
matter most before accepting low coverage there.
