# Glossary

Plain-English definitions of terms used across this repo and its docs.
Alphabetical. If you had to look something up while reading `CONTRIBUTING.md`
or `AGENTS.md` and it isn't here, add it.

**Branch protection** — a GitHub setting that stops anyone from pushing
directly to `main`; changes must go through a reviewed pull request instead.

**CI (continuous integration)** — automated checks (tests, linting) that run
on GitHub every time someone opens or updates a pull request, so problems
are caught before a human reviewer even looks at it. Configured in
`.github/workflows/ci.yml`.

**CODEOWNERS** — a file (`.github/CODEOWNERS`) listing who must approve
changes to which parts of the repo.

**Commit** — a saved snapshot of your changes, with a message describing
what and why. Several commits make up a pull request.

**Complexity (cyclomatic complexity) / complexity grade** — a score for how
many different paths a function can take (from `if`/`else`, loops, etc.).
More paths means more cases to think through and test, so this project caps
functions at grade A (the simplest tier) — see `docs/conventions/testing.md`
and `docs/conventions/function-design.md`.

**Config / YAML** — settings for a run, written in `.yaml` files under
`configs/` (e.g. `configs/default.yaml`). YAML is a plain-text format for
key/value settings — no code required to edit it.

**Cycle shape / density shape (forecasting)** — the shape of demand over a
sales cycle (e.g. a Natura campaign period), expressed as a distribution
(density) rather than a single total — how volume is spread across the days
of the cycle, per sector, not just how much volume there is in total.

**Formulation** — a specific way of mathematically expressing a problem so a
solver can work on it. Comparing formulations means trying different
mathematical setups of the same underlying problem.

**INA (anomaly detection)** — the project's term for flagging unusual/outlier
demand patterns (as opposed to forecasting the "normal" volume or cycle
shape). See `docs/literature/README.md`'s Zotero collection tree.

**Lagrangian relaxation** — an exact-optimization technique that relaxes
some hard constraints into the objective (with penalty multipliers), solving
an easier problem repeatedly while adjusting the multipliers to bound or
reach the original problem's optimum.

**Lint / linting** — automated checks for likely bugs and style issues
(unused variables, wrong types, etc.), done here by a tool called `ruff`.

**Metaheuristic** — a general-purpose search strategy (e.g. simulated
annealing, genetic algorithms) that explores good-enough solutions without
guaranteeing optimality, used when exact methods are too slow for the problem
size. See **SA**.

**MIP (mixed-integer program)** — an optimization problem where some
decision variables must be whole numbers (e.g. "use this route or don't," not
"use 30% of this route"). Solved here via `src/solver/mip/`.

**Objective** — the number a solver is trying to make as large or small as
possible (e.g. minimize cost, maximize coverage).

**Pre-commit hook** — a script that runs automatically right before `git
commit` completes, and can stop the commit if something's wrong. This
project's is `scripts/hooks/pre-commit`; see `CONTRIBUTING.md` for what it
checks.

**Pull request (PR)** — a request to merge your branch's changes into
`main`, reviewed by a teammate before it's accepted.

**Repository (repo)** — the project's folder plus its full history, tracked
by git.

**SA (simulated annealing)** — a metaheuristic that searches for good
solutions by accepting occasional worse moves (with a probability that
shrinks over time, like cooling metal) so it can escape local optima rather
than getting stuck near its starting point.

**Solver** — the software that actually searches for a solution to an
optimization problem once it's set up as a model (e.g. Gurobi, HiGHS,
OR-Tools CP-SAT).

**Staging (`git add`)** — marking specific changed files as "include these
in my next commit," as opposed to every changed file in the repo.
