"""Testes do parsing e da validação dos YAMLs do experimento sa_calibration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.calibration.config import (
    Dimension,
    SACalibrationConfig,
    load_calibration_config,
)

CONFIG_DIR = Path("configs/experiments/sa_calibration")


@pytest.mark.parametrize("name", ["full.yaml", "smoke.yaml"])
def test_shipped_configs_parse(name):
    config = load_calibration_config(CONFIG_DIR / name)

    assert config.description.status in {"exploratory", "in-progress", "confirmed"}
    assert config.budget.max_iterations > 0
    assert set(config.validate_phase.reference_configs) <= set(config.named_configs)


@pytest.fixture(scope="module")
def full():
    return load_calibration_config(CONFIG_DIR / "full.yaml")


def test_full_config_uses_the_fixed_budget_and_highs(full):
    assert full.budget.max_iterations == 200_000
    assert full.mip.solver == "appsi_highs"


def test_full_config_revalidates_with_new_seeds(full):
    assert len(full.screen.seeds) >= 5
    assert set(full.validate_phase.seeds).isdisjoint(full.screen.seeds)


def test_full_config_revalidates_on_an_unseen_tight_instance(full):
    unseen = set(full.validate_phase.instances) - set(full.screen.instances)

    assert unseen
    assert all(full.instances[name].tight for name in unseen)


def test_rejects_phase_with_unknown_instance():
    raw = yaml.safe_load((CONFIG_DIR / "smoke.yaml").read_text())
    raw["screen"]["instances"].append("nope")

    with pytest.raises(ValidationError, match="nope"):
        SACalibrationConfig.model_validate(raw)


@pytest.mark.parametrize(
    "bounds", [{"low": 2.0, "high": 1.0}, {"low": 0.0, "high": 1.0, "log": True}]
)
def test_rejects_invalid_dimension(bounds):
    with pytest.raises(ValidationError):
        Dimension(**bounds)
