# Reference for implemented level features

**Date:** 2026-10-06

## Task

Create one reference listing every currently supported level feature and how
to use it, choosing an appropriate documentation location.

## Outcome

Added a Portuguese feature catalog under `docs/guides`, with automatic features,
configurable families, organization categories and complete YAML examples.

## What changed

`docs/guides/forecasting-level-features.md` documents the source functions,
defaults, feature counts, auxiliary columns, ratio target and multi-step limits.
The organization guide links to the catalog, and this entry is indexed.

## Notes

The catalog distinguishes earlier observations from calendar lags, and model
parameters from predictors. Code and experiment configurations are unchanged.
