"""Testes da instrumentação do SA: callback, métricas novas do resultado e agenda por orçamento."""

from __future__ import annotations

import math
import random

import pytest

from src.solver.heuristics.simulated_annealing import (
    AnnealingParams,
    AnnealingResult,
    cooling_rate_for_budget,
    iterations_for_schedule,
    solve,
)

BUDGET = 500
T0 = 100.0
T_MIN = 0.1


def _no_stagnation(**overrides) -> AnnealingParams:
    return AnnealingParams(
        max_iterations=BUDGET, stagnation_window=BUDGET, stagnation_tolerance=0.0, **overrides
    )


def test_two_sector_instance_reaches_zero_amplitude(sa_instance):
    result = solve(**sa_instance, params=_no_stagnation(), rng=random.Random(1))

    assert result.objective == 0.0
    assert result.zmax == result.zmin == 10.0
    assert result.std_load == 0.0
    assert result.feasible


def test_zmax_minus_zmin_is_the_objective(sa_instance):
    as_is_only = _no_stagnation(initial_temperature=1.0, min_temperature=2.0)  # 0 iterações

    result = solve(**sa_instance, params=as_is_only, rng=random.Random(1))

    assert result.iterations == 0
    assert (result.zmax, result.zmin) == (20.0, 0.0)
    assert result.zmax - result.zmin == pytest.approx(result.objective)
    assert result.std_load == pytest.approx(10.0)


def test_time_is_measured(sa_instance):
    result = solve(**sa_instance, params=_no_stagnation(), rng=random.Random(1))

    assert result.time > 0.0


def test_callback_sees_every_iteration(sa_instance):
    calls = []

    def callback(*, iteration, energy, temperature, assignment):
        calls.append((iteration, energy, temperature, assignment.copy()))

    result = solve(**sa_instance, params=_no_stagnation(), rng=random.Random(1), callback=callback)

    assert [c[0] for c in calls] == list(range(result.iterations))
    temperatures = [c[2] for c in calls]
    assert temperatures == sorted(temperatures, reverse=True)
    assert all(c[3].shape == (2,) for c in calls)


def test_callback_does_not_change_the_search(sa_instance):
    params = _no_stagnation()
    without = solve(**sa_instance, params=params, rng=random.Random(7))
    with_cb = solve(**sa_instance, params=params, rng=random.Random(7), callback=lambda **_: None)

    assert (without.assignment, without.energy, without.iterations) == (
        with_cb.assignment,
        with_cb.energy,
        with_cb.iterations,
    )


@pytest.mark.parametrize(
    ("capacity_penalty", "churn_penalty", "expected"),
    [(0.0, 0.0, True), (1.0, 0.0, False), (0.0, 2.0, False)],
)
def test_feasible_requires_both_penalties_at_zero(capacity_penalty, churn_penalty, expected):
    result = AnnealingResult({}, 0.0, 0.0, capacity_penalty, churn_penalty, 0, "max_iterations")

    assert result.feasible is expected


def test_cooling_rate_for_budget_reaches_t_min_at_the_budget():
    alpha = cooling_rate_for_budget(BUDGET, T0, T_MIN)

    assert T0 * alpha**BUDGET == pytest.approx(T_MIN)


def test_iterations_for_schedule_inverts_cooling_rate_for_budget():
    alpha = cooling_rate_for_budget(BUDGET, T0, T_MIN)

    assert abs(iterations_for_schedule(T0, T_MIN, alpha) - BUDGET) <= 1


def test_iterations_for_schedule_matches_the_geometric_formula():
    expected = math.ceil(math.log(0.01 / 1000.0) / math.log(0.99))

    assert iterations_for_schedule(1000.0, 0.01, 0.99) == expected


def test_budget_matched_run_spends_the_whole_budget_without_stagnating(sa_instance):
    alpha = cooling_rate_for_budget(BUDGET, T0, T_MIN)
    params = _no_stagnation(initial_temperature=T0, min_temperature=T_MIN, cooling_rate=alpha)

    result = solve(**sa_instance, params=params, rng=random.Random(3))

    assert result.stop_reason != "stagnation"
    assert abs(result.iterations - BUDGET) <= 1
