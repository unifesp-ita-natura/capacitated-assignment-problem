# Update the forecasting branch from remote main

**Date:** 2026-10-30

## Task

Bring remote main into the current forecasting branch while preserving local
hierarchy experiments and the level/shape combination. Keep optimization and
simulated annealing exactly as provided by remote main.

## Outcome

Fetched origin/main at `6d64449` and fast-forwarded `codex/hierarchical-forecast`
from `e1cd941`. Main already contains the combined forecast integration.
Local tracked edits were protected in a named stash and reapplied; new hierarchy
files were copied to an ignored local backup before the update.

## What changed

Updated hierarchy experiment imports to use `src.forecasting.harness` and
`src.persistence.provenance`, following main's relocation of the old records
module. Resolved the agent-log index conflict by retaining both histories.
Ignored root-level `outputs/` so presentation files and their dependency
junction remain local. No solver, calibration, scaling or optimization
pipeline source was edited locally.

## Notes

The local hierarchy changes remain uncommitted; no remote push was requested.
The backup is `.venv/main_update_backup_2026-10-30/`, and the named pre-update
stash is retained until the user decides to remove it. The update preserves
the existing combined pipeline; it does not switch that pipeline to a new
hierarchical level model.

Validation: 66 focused tests passed across hierarchy, combined level/shape
forecasting and the shared run harness. Ruff formatting/lint passed and both
adapted executors remain Radon A. Git confirmed no local differences against
origin/main in solver, calibration, scaling, pipeline, main entry point or
combined forecast implementation. Hashes of the 32 backed-up new files
confirmed that only the two intended experiment import adaptations changed.

The broader pipeline test could not be collected because this local virtual
environment lacks Pyomo; no optimizer test or real solver was run. The local
Python executable also requires execution outside the sandbox on this machine;
the focused checks were run with the user's permission. No dependencies were
installed or updated during this operation.
