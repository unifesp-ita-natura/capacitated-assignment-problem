"""Testes da energia do SA nos dois modos: "range" (amplitude) e "peak_std_breach"."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.solver.heuristics.simulated_annealing import (
    AnnealingParams,
    _build_problem,
    _evaluate,
    solve,
)
from src.solver.inputs import ProblemInstance
from tests.solver.heuristics.test_simulated_annealing import _instance

_CAPACITY = 40.0


def _random_instance() -> ProblemInstance:
    """6 setores, 4 combinações, 2 ciclos de 4 dias, 2 CDs, capacidade que chega a estourar."""
    sectors, combos, days = range(1, 7), range(1, 5), range(1, 9)
    demand = _random_demand(sectors, days, combos)
    return ProblemInstance(
        sectors=list(sectors),
        combinations=list(combos),
        days=list(days),
        cd_sectors={1: [1, 2, 3], 2: [4, 5, 6]},
        daily_capacity={(c, a): _CAPACITY for c in (1, 2) for a in days},
        projected_demand=demand,
        current_assignment={s: 1 for s in sectors},
        max_churn=2,
        days_by_cycle={0: [1, 2, 3, 4], 1: [5, 6, 7, 8]},
    )


def _problem(instance: ProblemInstance):
    return _build_problem(
        instance.sectors,
        instance.combinations,
        instance.days,
        instance.cd_sectors,
        instance.daily_capacity,
        instance.projected_demand,
        instance.current_assignment,
        instance.max_churn,
        instance.days_by_cycle,
        None,
    )


def _daily_totals(instance: ProblemInstance, assignment: dict[int, int]) -> dict[int, float]:
    return {
        a: sum(instance.projected_demand[s, a, assignment[s]] for s in instance.sectors)
        for a in instance.days
    }


def _manual_penalties(instance: ProblemInstance, assignment: dict[int, int]) -> float:
    excess = sum(
        max(
            0.0,
            sum(instance.projected_demand[s, a, assignment[s]] for s in members)
            - instance.daily_capacity[c, a],
        )
        for c, members in instance.cd_sectors.items()
        for a in instance.days
    )
    changed = sum(assignment[s] != instance.current_assignment[s] for s in instance.sectors)
    return excess + max(0.0, 2.0 * changed - 2.0 * instance.max_churn)


def _random_demand(sectors, days, combos) -> dict[tuple[int, int, int], float]:
    rng = np.random.default_rng(7)
    keys = [(s, a, d) for s in sectors for a in days for d in combos]
    return {key: float(rng.uniform(0.0, 20.0)) for key in keys}


def _cycle_stats(totals: dict[int, float], instance: ProblemInstance) -> tuple[float, float, float]:
    """(pico, vale, desvio-padrão populacional), cada um em média entre os ciclos."""
    per_cycle = [_cycle_totals(totals, days) for days in instance.days_by_cycle.values()]
    return tuple(float(np.mean([stat(c) for c in per_cycle])) for stat in (np.max, np.min, np.std))


def _cycle_totals(totals: dict[int, float], days) -> np.ndarray:
    return np.array([totals[a] for a in days])


# Trajetória gravada com o SA anterior à introdução de `energy_mode` (commit ef55716),
# na mesma instância, parâmetros e seeds: o modo "range" tem de reproduzi-la bit a bit.
_LEGACY_PARAMS = AnnealingParams(
    initial_temperature=50.0,
    cooling_rate=0.999,
    min_temperature=0.01,
    max_iterations=3000,
    penalty_coefficient=5.0,
    stagnation_window=3000,
    stagnation_tolerance=0.0,
)
_LEGACY_BEST = {1: 1, 2: 1, 3: 1, 4: 3, 5: 3, 6: 3}
_LEGACY = {  # seed -> (energia da melhor, soma das energias correntes de todas as iterações)
    1: (21.415885017813874, 112869.43975767231),
    2: (21.415885017813874, 118613.49758776359),
}


@pytest.mark.parametrize("seed", sorted(_LEGACY))
def test_range_mode_reproduces_the_legacy_trajectory(seed):
    # A energia final é recalculada do zero e bate bit a bit; a soma das energias
    # correntes acumula o arredondamento da avaliação incremental (~1e-16 relativo).
    # Que os movimentos são idênticos está em test_simulated_annealing_incremental.py.
    energies: list[float] = []

    result = solve(
        _random_instance(),
        params=_LEGACY_PARAMS,
        rng=random.Random(seed),
        callback=lambda **kw: energies.append(kw["energy"]),
    )

    best_energy, energy_sum = _LEGACY[seed]
    assert result.energy == best_energy
    assert sum(energies) == pytest.approx(energy_sum, rel=1e-12)
    assert result.assignment == _LEGACY_BEST


def test_range_energy_is_amplitude_plus_rho_times_penalties():
    instance = _random_instance()
    assignment = {1: 2, 2: 4, 3: 1, 4: 3, 5: 1, 6: 2}
    rho = 3.5
    peak, valley, _ = _cycle_stats(_daily_totals(instance, assignment), instance)

    ev = _evaluate(
        _problem(instance),
        np.array([assignment[s] - 1 for s in range(1, 7)]),
        AnnealingParams(penalty_coefficient=rho),
    )

    expected = (peak - valley) + rho * _manual_penalties(instance, assignment)
    assert ev.energy == pytest.approx(expected, rel=1e-12)
    assert ev.objective == pytest.approx(peak - valley, rel=1e-12)


def test_peak_std_energy_is_the_weighted_sum():
    instance = _random_instance()
    assignment = {1: 2, 2: 4, 3: 1, 4: 3, 5: 1, 6: 2}
    params = AnnealingParams(
        energy_mode="peak_std_breach", w_peak=1.0, w_std=2.5, w_range=0.3, penalty_coefficient=7.0
    )
    peak, valley, std = _cycle_stats(_daily_totals(instance, assignment), instance)

    ev = _evaluate(_problem(instance), np.array([assignment[s] - 1 for s in range(1, 7)]), params)

    balance = 1.0 * peak + 2.5 * std + 0.3 * (peak - valley)
    assert ev.energy == pytest.approx(balance + 7.0 * _manual_penalties(instance, assignment))
    assert ev.objective == pytest.approx(peak - valley)  # o gap continua medido na amplitude


@pytest.mark.parametrize(
    "params",
    [
        AnnealingParams(max_iterations=500),
        AnnealingParams(max_iterations=500, energy_mode="peak_std_breach"),
    ],
    ids=["range", "peak_std_breach"],
)
def test_two_sectors_level_to_zero_amplitude_in_both_modes(params):
    result = solve(_instance(), rng=random.Random(1), params=params)

    assert result.objective == 0.0
    assert result.zmax == 10.0
    assert result.std_load == 0.0


def test_defaults_are_unchanged():
    params = AnnealingParams()

    assert params.energy_mode == "range"
    assert (params.w_peak, params.w_std, params.w_range) == (1.0, 1.0, 0.0)
    assert (
        params.initial_temperature,
        params.cooling_rate,
        params.min_temperature,
        params.max_iterations,
        params.stagnation_window,
        params.stagnation_tolerance,
        params.sector_bias,
        params.destination_bias,
        params.penalty_coefficient,
    ) == (1000.0, 0.99, 0.01, 100_000, 200, 1e-5, 0.8, 0.7, 1e6)


def test_unknown_energy_mode_is_rejected():
    with pytest.raises(ValueError, match="energy_mode"):
        AnnealingParams(energy_mode="peak")
