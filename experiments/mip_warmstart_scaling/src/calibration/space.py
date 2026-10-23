"""Espaço de busca do SA: amostragem LHS (fase 1) e sugestões TPE do Optuna (fase 2)."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence

import optuna
from scipy.stats import qmc

from src.calibration.config import Dimension

Config = dict[str, float]


def scale_unit(dimension: Dimension, unit: float) -> float:
    """Leva `unit` em [0, 1] ao intervalo da dimensão (log10 se `dimension.log`)."""
    if dimension.log:
        low, high = math.log10(dimension.low), math.log10(dimension.high)
        return float(10 ** (low + unit * (high - low)))
    return float(dimension.low + unit * (dimension.high - dimension.low))


def lhs_configs(space: Mapping[str, Dimension], n_configs: int, seed: int) -> list[Config]:
    """`n_configs` pontos de um hipercubo latino sobre o espaço, na ordem de `space`."""
    names = list(space)
    unit_points = qmc.LatinHypercube(d=len(names), rng=seed).random(n_configs)
    return [
        {name: scale_unit(space[name], u) for name, u in zip(names, point, strict=True)}
        for point in unit_points
    ]


def distributions(
    space: Mapping[str, Dimension],
) -> dict[str, optuna.distributions.FloatDistribution]:
    """As distribuições do Optuna equivalentes ao espaço de busca."""
    return {
        name: optuna.distributions.FloatDistribution(dim.low, dim.high, log=dim.log)
        for name, dim in space.items()
    }


def create_study(seed: int) -> optuna.Study:
    """Estudo de minimização com TPE multivariado e `constant_liar` (lotes paralelos)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():  # multivariate/constant_liar são "experimentais" no Optuna
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, constant_liar=True)
    return optuna.create_study(direction="minimize", sampler=sampler)


def warm_start(
    study: optuna.Study,
    space: Mapping[str, Dimension],
    history: Sequence[tuple[Config, float]],
) -> None:
    """Registra no estudo configurações já avaliadas (config, score) como trials completos."""
    dists = distributions(space)
    for config, score in history:
        params = {name: config[name] for name in space}
        study.add_trial(optuna.trial.create_trial(params=params, distributions=dists, value=score))


def ask_batch(study: optuna.Study, space: Mapping[str, Dimension], size: int) -> list[Config]:
    """Pede `size` novas configurações ao TPE (sem avaliá-las)."""
    dists = distributions(space)
    return [dict(study.ask(dists).params) for _ in range(size)]
