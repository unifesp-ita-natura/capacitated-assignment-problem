"""Load a RunConfig from a YAML file."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.config.schema import RunConfig


def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return RunConfig.model_validate(raw)
