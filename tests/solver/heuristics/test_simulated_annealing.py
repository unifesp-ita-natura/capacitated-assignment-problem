"""Testes do Simulated Annealing contra instâncias pequenas de ótimo conhecido."""

from __future__ import annotations

import random

import pytest

from src.solver.heuristics.simulated_annealing import AnnealingParams, solve

# Instância base: 2 setores, 2 combinações, 2 dias, 1 CD.
# Combinação 1 -> toda a demanda cai no dia 1; combinação 2 -> tudo no dia 2.
# As-Is coloca os dois setores na combinação 1: dia 1 = 20, dia 2 = 0 (amplitude 20).
# Ótimo: um setor em cada combinação -> dia 1 = 10, dia 2 = 10 (amplitude 0).
_DEMAND = {
    (1, 1, 1): 10.0,
    (1, 2, 1): 0.0,
    (1, 1, 2): 0.0,
    (1, 2, 2): 10.0,
    (2, 1, 1): 10.0,
    (2, 2, 1): 0.0,
    (2, 1, 2): 0.0,
    (2, 2, 2): 10.0,
}


def _instance(**overrides) -> dict:
    base = dict(
        sectors=[1, 2],
        combinations=[1, 2],
        days=[1, 2],
        cd_sectors={1: [1, 2]},
        daily_capacity={(1, 1): 100.0, (1, 2): 100.0},
        projected_demand=_DEMAND,
        current_assignment={1: 1, 2: 1},
        max_churn=2,
    )
    base.update(overrides)
    return base


def test_levels_demand_to_zero_amplitude():
    result = solve(**_instance(), rng=random.Random(1), params=AnnealingParams(max_iterations=500))

    assert result.objective == 0.0
    assert set(result.assignment.values()) == {1, 2}


def test_never_returns_worse_than_as_is():
    result = solve(**_instance(), rng=random.Random(1))

    # As-Is tem energia 20 (amplitude 20, sem penalidades); o SA nunca piora isso.
    assert result.energy <= 20.0


def test_respects_zero_churn_budget():
    result = solve(
        **_instance(max_churn=0), rng=random.Random(1), params=AnnealingParams(max_iterations=500)
    )

    assert result.assignment == {1: 1, 2: 1}
    assert result.churn_penalty == 0.0
    assert result.objective == 20.0


def test_fixes_capacity_violation():
    tight = _instance(daily_capacity={(1, 1): 10.0, (1, 2): 100.0})

    result = solve(**tight, rng=random.Random(1), params=AnnealingParams(max_iterations=500))

    assert result.capacity_penalty == 0.0


def test_is_deterministic_for_a_fixed_seed():
    first = solve(**_instance(), rng=random.Random(42))
    second = solve(**_instance(), rng=random.Random(42))

    assert first.assignment == second.assignment
    assert first.energy == second.energy


def test_stop_reason_is_one_of_the_known_values():
    result = solve(**_instance(), rng=random.Random(1))

    assert result.stop_reason in {"min_temperature", "stagnation", "max_iterations"}


def test_valid_combinations_restricts_the_domain():
    # O CD 1 só pode usar a combinação 1 -> nenhum setor consegue sair dela.
    result = solve(
        **_instance(valid_combinations={1: [1]}),
        rng=random.Random(1),
        params=AnnealingParams(max_iterations=500),
    )

    assert result.assignment == {1: 1, 2: 1}


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_finds_the_optimum_across_seeds(seed):
    result = solve(
        **_instance(), rng=random.Random(seed), params=AnnealingParams(max_iterations=500)
    )

    assert result.objective == 0.0
