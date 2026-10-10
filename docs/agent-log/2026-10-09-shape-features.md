# Organization and calendar features for shape

**Date:** 2026-10-09

## Task

Add region, sales management, weekday and month predictors to the isolated
shape pipeline. Holiday implementation was deferred by the user.

## Outcome

A new pooled LightGBM candidate learns normalized daily shape intensities.
An example YAML compares its basic and feature-augmented versions with the
existing uniform and historical-sector curves, without changing level.

## What changed

Added `shape_learning.py`, feature tests, a reference guide and a feature
comparison YAML. Shape data preserves organization codes; the evaluation
dispatches the new candidate and archives effective features and parameters.

## Notes

Training alone defines categorical vocabularies and historical organization
carry-forward; holdout assignments and demand are excluded from predictors.
The native LightGBM API uses no scikit-learn dependency. Negative intensities
are clipped and all curves normalized to sum to one, with uniform fallback.
Holiday code/templates drafted during clarification were removed after the
user deferred that feature.

Validation: the combined shape/dashboard suite passed 62 tests before the final
additional archive/organization checks; the final targeted suite passed all
39 shape/feature tests without warnings. Ruff passed and added functions have
Radon grade A. A real-base smoke evaluation used 30 training cycles, one held-out
cycle and 10 boosting rounds, scoring 612 sector curves per candidate; all curve
shares summed to one within 1.11e-16. This smoke check created no experiment
folder and does not replace the full feature comparison configured in YAML.
