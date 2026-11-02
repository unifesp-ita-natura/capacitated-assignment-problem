"""Testes do Relax-and-Fix resolvendo de verdade com o HiGHS, em instâncias minúsculas."""

from __future__ import annotations

import pyomo.environ as pyo
import pytest

from src.solver.inputs import ProblemInstance
from src.solver.matheuristics import RelaxAndFixParams, solve
from src.solver.matheuristics.evaluation import evaluate_assignment
from src.solver.matheuristics.problem import build_problem
from src.solver.matheuristics.subproblem import (
    apply_variable_modes,
    pyomo_subproblem_solver,
    read_assignment,
)
from src.solver.mip.block_assignment import build_block_assignment_model

pytest.importorskip("highspy")


def _demand(sectors: list[int]) -> dict[tuple[int, int, int], float]:
    return {**{(s, 1, 1): 10.0 for s in sectors}, **{(s, 2, 2): 10.0 for s in sectors}}


def _capacity(per_day: float) -> dict[tuple[int, int], float]:
    return {(cd, day): per_day for cd in (1, 2) for day in (1, 2)}


def _instance(capacity: float = 1000.0, max_churn: float = 4) -> ProblemInstance:
    """4 setores em 2 CDs; combinação 1 -> 10 itens no dia 1, combinação 2 -> 10 no dia 2.

    O As-Is põe todo mundo na combinação 1 (amplitude 40); o ótimo divide 2 e 2 (amplitude 0).
    """
    sectors = [1, 2, 3, 4]
    return ProblemInstance(
        sectors=sectors,
        combinations=[1, 2],
        days=[1, 2],
        cd_sectors={1: [1, 2], 2: [3, 4]},
        daily_capacity=_capacity(capacity),
        projected_demand=_demand(sectors),
        current_assignment={s: 1 for s in sectors},
        max_churn=max_churn,
        days_by_cycle={0: [1, 2]},
    )


def _mip_optimum(instance: ProblemInstance) -> float:
    model = build_block_assignment_model(instance)
    pyo.SolverFactory("appsi_highs").solve(model)
    return pyo.value(model.objective)


def test_relax_and_fix_reaches_the_mip_optimum_on_a_toy_instance():
    instance = _instance()

    result = solve(instance, params=RelaxAndFixParams(window=1, step=1))

    assert (result.feasible, result.objective) == (True, _mip_optimum(instance))


def test_relax_and_fix_repairs_an_as_is_that_exceeds_capacity():
    instance = _instance(capacity=15.0)
    problem = build_problem(instance)
    as_is = evaluate_assignment(problem, problem.as_is)

    result = solve(instance, params=RelaxAndFixParams(window=1, step=1))

    # O As-Is carrega 20 por CD no dia 1 (capacidade 15); a solução devolvida cabe.
    assert (as_is.feasible, result.feasible) == (False, True)


def test_persistent_solver_picks_up_changes_of_the_fixed_variables():
    model = build_block_assignment_model(_instance())
    subproblem_solver = pyomo_subproblem_solver("appsi_highs")

    apply_variable_modes(model, fixed={1: 1, 2: 1, 3: 1, 4: 1}, binary=set())
    first_solved = subproblem_solver(model, 30.0)
    first = read_assignment(model, [1, 2, 3, 4])
    apply_variable_modes(model, fixed={1: 2, 2: 2, 3: 2, 4: 2}, binary=set())
    second_solved = subproblem_solver(model, 30.0)
    second = read_assignment(model, [1, 2, 3, 4])

    assert (first_solved, second_solved) == (True, True)
    assert (set(first.values()), set(second.values())) == ({1}, {2})


def test_subproblem_with_an_infeasible_fixing_reports_no_solution():
    model = build_block_assignment_model(_instance(max_churn=0))
    apply_variable_modes(model, fixed={1: 2, 2: 2, 3: 2, 4: 2}, binary=set())

    solved = pyomo_subproblem_solver("appsi_highs")(model, 30.0)

    # Mudar os 4 setores estoura o orçamento de churn (M = 0), então o subproblema é inviável.
    assert solved is False
