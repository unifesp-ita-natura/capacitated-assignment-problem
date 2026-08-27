# data

Where problem data lives. Nothing under here except this file (and the
`.gitkeep` placeholders) is committed to git — see "Why data isn't
committed" below, and the size check in the pre-commit hook that backs it up.

- `instances/` — raw problem instances (the actual optimization problems to
  solve), whether hand-written, downloaded, or generated.
- `processed/` — derived/cleaned versions of the above, produced by a script
  you can re-run — never edited by hand.
- `generated/` — synthetic instances produced by `src/generation/`-style
  code, keyed however that code chooses (e.g. one subdirectory per
  experiment, matching `configs/experiments/<name>/`).

Referenced from `configs/default.yaml`'s `paths:` block; point an
experiment config at a different location if it needs one.

## Why data isn't committed

Git is bad at storing large or frequently-regenerated binary/CSV/JSON data:
every version bloats the repo permanently (even after the file is deleted,
it's still in history), and diffs are meaningless for anything that isn't
plain text. Instead:

- **Generated/derived data** should be reproducible by re-running the
  generator/script against a config — commit the config and the code, not
  the output.
- **Raw data you can't regenerate** (e.g. a dataset someone downloaded or
  collected) needs a real home outside git: cloud storage, a shared drive,
  or your cluster's storage — and a short note here (or in the relevant
  experiment's `README.md`) on where to get it and how to place it under
  `data/`.
- **Small fixed fixtures for tests** (a handful of KB, used by the test
  suite itself) belong in `tests/` (e.g. `tests/fixtures/`), not here —
  test fixtures are code, not data, and should be reviewed like it.

The pre-commit hook (`scripts/hooks/pre-commit`) blocks committing any
staged file above 5 MB as a backstop — see its output for how to fix a
blocked commit (usually: add the file to `.gitignore` and store it
properly instead).
