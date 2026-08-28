"""Tests for config loading."""

from __future__ import annotations

from src.config import load_config


def test_load_default_config():
    config = load_config("configs/default.yaml")

    assert config.project_name == "capacitated-assignment-problem"
    assert config.solver == "gurobi"
    assert config.paths.results == "results"
