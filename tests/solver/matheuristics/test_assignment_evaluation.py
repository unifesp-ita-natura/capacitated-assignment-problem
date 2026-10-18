"""Testes da indexação do problema e da avaliação de atribuições (sem solver)."""

from __future__ import annotations

import pytest

from src.solver.matheuristics.evaluation import capacity_utilization, evaluate_assignment
from src.solver.matheuristics.problem import build_problem


def _problem(**overrides):
    """4 setores em 2 CDs; combinação 1 -> 10 itens no dia 1, combinação 2 -> 10 no dia 2."""
    demand = {}
    for sector in (1, 2, 3, 4):
        demand[(sector, 1, 1)] = 10.0
        demand[(sector, 2, 2)] = 10.0
    base = dict(
        sectors=[1, 2, 3, 4],
        cd_sectors={1: [1, 2], 2: [3, 4]},
        daily_capacity={(1, 1): 100.0, (1, 2): 100.0, (2, 1): 100.0, (2, 2): 100.0},
        projected_demand=demand,
        current_assignment={(s, 1): 1 for s in (1, 2, 3, 4)},
        max_churn=1,
        days_by_cycle={0: [1, 2]},
    )
    base.update(overrides)
    return build_problem(**base)


def test_build_problem_converts_the_sparse_as_is_to_sector_to_combination():
    problem = _problem(current_assignment={(1, 1): 1, (2, 2): 1, (3, 1): 0})

    assert problem.as_is == {1: 1, 2: 2}


def test_build_problem_maps_each_sector_to_its_cd():
    problem = _problem()

    assert problem.sector_cd == {1: 1, 2: 1, 3: 2, 4: 2}


def test_build_problem_rejects_a_sector_without_cd():
    with pytest.raises(ValueError, match="sem CD"):
        _problem(cd_sectors={1: [1, 2], 2: [3]})


def test_build_problem_rejects_a_sector_in_two_cds():
    with pytest.raises(ValueError, match="mais de um CD"):
        _problem(cd_sectors={1: [1, 2], 2: [2, 3, 4]})


def test_demand_index_drops_zero_quantities():
    problem = _problem(projected_demand={(1, 1, 1): 10.0, (1, 2, 1): 0.0})

    assert problem.demand_by_pair == {(1, 1): ((1, 10.0),)}


def test_objective_is_the_daily_range_of_the_assignment():
    problem = _problem()

    as_is = evaluate_assignment(problem, {1: 1, 2: 1, 3: 1, 4: 1})  # dia 1 = 40, dia 2 = 0
    balanced = evaluate_assignment(problem, {1: 1, 2: 1, 3: 2, 4: 2})  # 20 e 20

    assert as_is.objective == 40.0
    assert balanced.objective == 0.0


def test_objective_averages_the_range_over_cycles():
    demand = {(1, 1, 1): 10.0, (1, 3, 1): 30.0}
    problem = _problem(
        sectors=[1],
        cd_sectors={1: [1]},
        daily_capacity={},
        projected_demand=demand,
        current_assignment={(1, 1): 1},
        days_by_cycle={0: [1, 2], 1: [3, 4]},
    )

    # Ciclo 0: dia 1 = 10, dia 2 = 0 -> amplitude 10; ciclo 1: dia 3 = 30, dia 4 = 0 -> 30.
    assert evaluate_assignment(problem, {1: 1}).objective == 20.0


def test_capacity_violation_sums_the_excess_over_every_cd_and_day():
    problem = _problem(
        daily_capacity={(1, 1): 15.0, (1, 2): 100.0, (2, 1): 5.0, (2, 2): 100.0},
        max_churn=4,
    )

    evaluation = evaluate_assignment(problem, {1: 1, 2: 1, 3: 1, 4: 1})

    # CD 1 dia 1: 20 - 15 = 5; CD 2 dia 1: 20 - 5 = 15.
    assert evaluation.capacity_violation == 20.0
    assert not evaluation.feasible


def test_missing_capacity_entry_counts_as_zero_like_the_mip():
    problem = _problem(daily_capacity={})

    # Capacidade 0 em todo (CD, dia): cada CD carrega 20 itens no dia 1.
    assert evaluate_assignment(problem, {1: 1, 2: 1, 3: 1, 4: 1}).capacity_violation == 40.0


def test_capacity_utilization_is_the_worst_load_over_capacity_ratio():
    problem = _problem(daily_capacity={(1, 1): 40.0, (1, 2): 100.0, (2, 1): 20.0, (2, 2): 100.0})

    # Tudo na combinação 1: CD 1 e CD 2 carregam 20 no dia 1 -> razões 0,5 e 1,0.
    assert capacity_utilization(problem, {1: 1, 2: 1, 3: 1, 4: 1}) == 1.0


def test_capacity_utilization_ignores_pairs_without_capacity():
    problem = _problem(daily_capacity={(1, 1): 0.0, (1, 2): 100.0, (2, 1): 50.0, (2, 2): 100.0})

    assert capacity_utilization(problem, {1: 1, 2: 1, 3: 1, 4: 1}) == 0.4


def test_churn_budget_charges_two_per_changed_sector():
    problem = _problem(max_churn=1)

    unchanged = evaluate_assignment(problem, {1: 1, 2: 1, 3: 1, 4: 1})
    one_change = evaluate_assignment(problem, {1: 2, 2: 1, 3: 1, 4: 1})
    two_changes = evaluate_assignment(problem, {1: 2, 2: 2, 3: 1, 4: 1})

    assert (unchanged.churn_budget_used, one_change.churn_budget_used) == (0.0, 2.0)
    assert one_change.feasible
    assert two_changes.churn_budget_used == 4.0
    assert not two_changes.feasible


def test_sector_without_as_is_charges_one():
    problem = _problem(current_assignment={(1, 1): 1})

    assert evaluate_assignment(problem, {1: 1, 2: 1}).churn_budget_used == 1.0
