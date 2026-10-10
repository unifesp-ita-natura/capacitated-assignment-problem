# Simple level feature-engineering manual

**Date:** 2026-10-07

## Task

Give an intern a simple Windows manual for environment setup, dependency
installation, YAML configuration and running feature experiments.

## Outcome

A Portuguese guide covers first-time setup and the repeatable workflow for
testing an existing feature against an otherwise identical LightGBM baseline.

## What changed

Added `docs/guides/manual-simples-feature-engineering-level.md` with commands,
a complete configuration, result interpretation and common errors. Included
a separate section for new code features and relevant quality checks.

## Notes

Uses a minimal forecasting environment and direct `.venv` commands for cmd.
The user's existing YAMLs and Python files are unchanged. Documentation-only
validation checks local links and configuration structure; no backtest is run.
