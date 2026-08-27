# Agent Log Convention

## Purpose

Researchers need a record of what AI agents did in this repo, beyond what
commit messages capture — the reasoning, the paths tried and abandoned, and
context that helps a human pick the work back up later. Entries also serve
as the raw material for reporting on agent activity (e.g. summarizing a
week's or a project's worth of agent work for a supervisor or paper),
so write for a reader who wasn't in the session, not just for the next
agent.

## Where entries live

`docs/agent-log/` — one file per entry, named:

```text
YYYY-MM-DD-short-slug.md
```

Examples: `2026-08-14-agent-log-setup.md`, `2026-08-20-solver-timeout-fix.md`.

One file per distinct piece of work (a session, a feature, a bug
investigation), not one file per day. Dated filenames keep entries
sortable and diffable in git history; a single append-only doc is harder to
search and tends to grow unbounded.

## Entry format

```markdown
# <title>

**Date:** YYYY-MM-DD
**Related:** <commit hash(es), PR link, or issue — optional>

## Task

What was asked, and why (the goal or problem behind the request).

## Outcome

One or two sentences a non-technical reader could lift into a status
report: what's now true that wasn't before.

## What changed

Files/modules touched and why, in a sentence or two — not a diff restatement.

## Notes

Anything non-obvious: approaches tried and rejected, open questions,
follow-ups for a human.
```

Keep entries short. If the "why" is fully captured by the commit message,
link the commit instead of repeating it.

## Index

`docs/agent-log/README.md` lists every entry, newest first, one line each:

```markdown
- [YYYY-MM-DD short title](YYYY-MM-DD-short-slug.md) — one-line summary
```

Update the index in the same commit that adds the entry.

## When to add an entry

Add an entry for work a human would want to trace later: nontrivial
features, bug investigations, refactors with tradeoffs, or anything where
the reasoning matters more than the diff. Skip it for trivial or
self-explanatory changes (typo fixes, formatting, dependency bumps).
