# AGENTS.md

## Purpose

This repository uses a consistent Git workflow for staging and committing changes. Keep commits focused, review the staged diff before committing, and use clear commit types. Full format, types, and staging rules: `docs/conventions/commits.md`.

## One-time setup

Run `make install-hooks` (or `scripts/install-hooks.sh`) once per clone. It
installs a `pre-commit` git hook that runs `ruff format --check`,
`ruff check`, and a `radon` complexity gate (grade A required) on staged
Python files — see `scripts/hooks/pre-commit`. Without it, formatting/lint/
complexity issues aren't caught until CI.

## Preferred behavior for agents

When making changes in this repository, agents should:

* verify the pre-commit hook is installed (`test -x .git/hooks/pre-commit`) and run `make install-hooks` if it isn't, before the first commit in a fresh clone or worktree
* stage only the files relevant to the requested task
* review staged changes before committing (`git status`, `git diff`, `git diff --staged`)
* choose the most accurate commit type and split unrelated work into separate commits — see `docs/conventions/commits.md`
* avoid creating noisy or catch-all commits
* ask the user for clarification whenever any requirement, instruction, or expected behavior is unclear
* keep tests independent from direct solver execution by mocking solver calls — see `docs/conventions/testing.md`
* ensure docs cover the supported workflow for running with the real solver
* document changes in `docs/` as the implementation evolves when the work affects documented behavior, workflows, or architecture, creating new `.md` files there when needed
* add an entry to `docs/agent-log/` for nontrivial work (features, bug investigations, refactors with tradeoffs) and update `docs/agent-log/README.md`'s index in the same commit — see `docs/conventions/agent-log.md`
* add or update the matching Lean file under `formal/` and compile it whenever a theorem/proposition/lemma in `docs/papers/` is added or changed — see `formal/README.md` (drop this line if the project doesn't use `formal/`)
* keep Python module headers short and current when editing tracked `.py` files — see `docs/conventions/module-headers.md`
* run `uv run ruff format ...`, `uv run ruff check ...`, and relevant `uv run pytest ...` commands after broad Python edits
* run `uv run radon cc -s -a ...` after every code addition; code must stay in complexity grade A
* follow the risk-based coverage targets and `pytest --cov=src` workflow in `docs/conventions/testing.md`
* follow `docs/conventions/function-design.md` when writing or refactoring functions

## What to avoid

* Vague commit messages: `update stuff`, `fix issues`, `misc changes`, `wip`.
* Combining unrelated fixes and features in the same commit.
* Committing unreviewed staged changes.

## Related conventions

* `docs/conventions/commits.md` — commit message format, types, staging rules, examples by scenario
* `docs/conventions/module-headers.md` — Python module docstring convention
* `docs/conventions/function-design.md` — small, single-purpose function guidelines
* `docs/conventions/testing.md` — solver mocking, coverage targets, lint/complexity commands
* `docs/conventions/agent-log.md` — agent log entry format and naming convention
