# Contributing

This is a plain-English walkthrough of how to make a change to this repo,
written for anyone joining the team regardless of software engineering
background. `AGENTS.md` covers the same ground in short bullet points aimed
at AI coding agents — this doc is the human-friendly version. Unfamiliar
term? Check `docs/glossary.md`.

## 1. One-time setup

Install [uv](https://docs.astral.sh/uv/) (manages Python and dependencies
for you), then from the repo root:

```bash
make setup
```

This installs everything the project needs, sets up an automatic check that
runs before each commit, and runs the test suite so you know right away if
something's wrong — you should see something like `15 passed` and no
errors.

Run `make setup` again (or just `make install-hooks`) if you ever delete and
re-clone the repo, or create a new git worktree — none of this carries over
automatically.

## 2. Making a change

1. Make sure you're up to date: `git pull`.
2. Create a branch for your change: `git checkout -b your-name/short-description`.
3. Edit the files you need to.
4. Run the tests to make sure nothing broke: `uv run pytest -q`.
5. Stage and commit — see below.
6. Push your branch: `git push -u origin your-name/short-description`.
7. Open a pull request (PR) on GitHub and ask a teammate to review it.

## 3. Committing your change

```bash
git status          # see what you changed
git diff             # see the actual lines
git add <file>       # stage only the files for this one change
git commit -m "fix: short description of what changed"
```

Only stage files relevant to the one thing you're doing — if you touched
two unrelated things, make two commits. See `docs/conventions/commits.md`
for the exact message format and the list of `type` prefixes to choose from
(`feat`, `fix`, `docs`, ...).

### The commit will run some automatic checks

When you run `git commit`, a script runs automatically on the files you're
committing (this is what `make install-hooks` set up):

- It **blocks any file over 5MB**. This is almost always a sign a data file
  or a results file snuck into the commit — see `data/README.md` for where
  those actually belong. The error message tells you which file and how big
  it is.
- If you're committing a Jupyter notebook (`.ipynb`), it **strips the cell
  outputs** before committing and re-stages the cleaned version for you —
  no action needed, this just keeps run-specific data and image blobs out
  of git history.

Then, for Python files, it checks three things, in order, and stops your
commit if any of them fail:

1. **Formatting** — your code doesn't match the project's standard style.
   Fix: run `uv run ruff format <the file it names>`, then `git add` and
   commit again. This one fixes itself automatically — you don't need to
   understand what changed.
2. **Lint** — something looks like a likely bug (an unused import, an
   undefined name, etc). The error message tells you the file and line;
   fix what it describes.
3. **Complexity** — a function has too many branches (`if`/`else`,
   loops, etc.) packed into it to reason about safely. See below.

If you get stuck on any of these, that's normal — ask in the team channel
or flag it in your PR description rather than working around the check.

### Fixing a complexity error

You'll see something like:

```
src/solver/mip/example.py
    F 12:0 solve_thing - B (9)
pre-commit: functions above complexity grade A found — simplify before committing.
```

This means the function `solve_thing`, starting at line 12, has too many
decision points (grade `B`, complexity `9`) — beyond the project's target of
grade `A`. This isn't about the code being "wrong," it's a signal that the
function is doing too much at once and will be harder for someone else (or
you, in six months) to follow.

The most common fix: find a chunk of the function that has its own clear
purpose — often a nested `if` block or a loop body — and pull it out into
its own, separately named function. Give it a name that describes *what* it
does. `docs/conventions/function-design.md` has more detail and examples.
If you're not sure how to split a specific function, that's a good thing to
ask a teammate to pair on rather than guess.

## 4. Getting help

- `docs/glossary.md` — plain-English definitions of terms used in this repo
  and its docs (MIP, solver, CI, branch protection, complexity grade, ...).
- `docs/conventions/` — the specific rules referenced above, each in its own
  short file.
- If a command in this doc doesn't work as described, that's a bug in the
  docs — say so, don't assume you did something wrong.
