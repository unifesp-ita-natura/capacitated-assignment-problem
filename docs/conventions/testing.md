# Testing Conventions

Tests live in `tests/`. `conftest.py` provides shared fixtures.

## Solver

**Do not call the real solver (e.g. Gurobi) directly in tests.** Mock solver
interactions instead of requiring a live installation or license. When
documentation describes how to run the project with the real solver in a
real environment, keep those instructions clear and separate from the test
suite's mocked path. See `src/solver/README.md` for the mocking pattern used
by the example solver module.

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
