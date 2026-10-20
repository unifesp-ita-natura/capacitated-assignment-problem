"""Testes da amostragem LHS e das sugestões do TPE (Optuna) no espaço de busca."""

from __future__ import annotations

import math

import pytest

from src.calibration.config import Dimension

# o optuna não é dependência do projeto: sem ele (ex.: CI), o módulo inteiro é pulado
space = pytest.importorskip("src.calibration.space")

SPACE = {
    "initial_temperature": Dimension(low=10.0, high=1000.0, log=True),
    "sector_bias": Dimension(low=0.0, high=1.0),
}


def _inside(config) -> bool:
    return all(SPACE[name].low <= value <= SPACE[name].high for name, value in config.items())


def test_scale_unit_log_and_linear():
    assert space.scale_unit(SPACE["initial_temperature"], 0.5) == pytest.approx(100.0)
    assert space.scale_unit(SPACE["sector_bias"], 0.25) == pytest.approx(0.25)


def test_lhs_has_one_point_per_stratum_in_every_dimension():
    n = 8
    configs = space.lhs_configs(SPACE, n, seed=0)

    assert len(configs) == n and all(_inside(c) for c in configs)
    temperature_strata = {
        int(n * math.log10(c["initial_temperature"] / 10.0) / 2.0) for c in configs
    }
    bias_strata = {int(n * c["sector_bias"]) for c in configs}
    assert len(temperature_strata) == len(bias_strata) == n


def test_lhs_is_reproducible():
    assert space.lhs_configs(SPACE, 4, seed=3) == space.lhs_configs(SPACE, 4, seed=3)


def test_tpe_batch_after_warm_start_stays_in_bounds():
    study = space.create_study(seed=0)
    history = [(config, float(i)) for i, config in enumerate(space.lhs_configs(SPACE, 10, seed=1))]
    space.warm_start(study, SPACE, history)

    batch = space.ask_batch(study, SPACE, 3)

    assert len(study.trials) == 13
    assert len(batch) == 3 and all(_inside(c) for c in batch)
