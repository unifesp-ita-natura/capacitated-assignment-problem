# Commit Conventions

## Before committing

* Check the current status with `git status`.
* Review your changes with `git diff`.
* Stage only the files and hunks that belong to the same logical change.
* Verify what will be committed with `git diff --staged`.

## Staging rules

* Stage the smallest coherent unit of work.
* Do not mix refactors, bug fixes, formatting, and new features in one commit unless they are inseparable.
* Do not commit generated files, secrets, local environment changes, or debugging leftovers unless explicitly required.
* When in doubt, split the work into multiple commits.

## Commit message format

```text
<type>(<scope>): <short description>
```

* `type`: the kind of change (see "Allowed commit types" below).
* `scope`: the area of the codebase affected — e.g. `solver`, `config`, `io`, `docs`. Adjust scopes to match your package layout as it grows.

Examples:

```text
feat(solver): add iteration callback for solve progress
fix(io): handle missing values in instance loader
refactor(config): simplify default settings
docs(readme): add local solver setup notes
chore(config): update default experiment settings
test(solver): add coverage for infeasible cases
perf(solver): reduce repeated allocation during solves
build(deps): pin Pyomo and Gurobi dependencies
ci(github-actions): run pytest on pull requests
style(tests): format solver assertions consistently
```

## Allowed commit types

| Type | Use for |
|---|---|
| `feat` | a new user-facing or developer-facing feature |
| `fix` | a bug fix |
| `refactor` | internal code changes that do not change behavior |
| `docs` | documentation-only changes |
| `chore` | maintenance tasks that are not features, fixes, or refactors |
| `test` | adding or updating tests |
| `perf` | measurable performance improvements |
| `build` | build system or dependency changes affecting packaging/build output |
| `ci` | continuous integration or automation pipeline changes |
| `style` | formatting or non-functional stylistic cleanup only |

## Commit writing guidelines

* Keep the description short and specific.
* Use imperative mood: `add`, `fix`, `update`, `remove`.
* Start with a lowercase description after the colon.
* Do not end the subject line with a period.
* Keep the subject focused on what changed; one logical change per commit.

## Recommended workflow

```bash
git status
git diff
git add <file>
git diff --staged
git commit -m "feat: add configurable retry policy"
```

## What to avoid

* Vague messages: `update stuff`, `fix issues`, `misc changes`, `wip`.
* Combining unrelated fixes and features in the same commit.
* Committing unreviewed staged changes.
