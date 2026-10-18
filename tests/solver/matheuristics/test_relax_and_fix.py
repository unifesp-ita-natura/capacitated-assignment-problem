"""Testes do Relax-and-Fix + Fix-and-Optimize — o solver dos subproblemas é um roteiro simulado."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.persistence import SolveResult
from src.solver.matheuristics.relax_and_fix import (
    COMPLETED,
    NO_SOLUTION_WINDOW,
    RelaxAndFixParams,
    run_relax_and_fix,
    solve,
)


def _instance(n_cds: int = 2, sectors_per_cd: int = 2, **overrides) -> dict:
    """`sectors_per_cd` setores por CD; combinação 1 -> 10 itens no dia 1, 2 -> 10 no dia 2.

    O As-Is põe todo mundo na combinação 1 (dia 1 = 10 * n_setores, dia 2 = 0), então
    a amplitude inicial é 10 * n_setores e o ótimo (metade em cada combinação) é 0.
    """
    sectors = list(range(1, sectors_per_cd * n_cds + 1))
    base = dict(
        sectors=sectors,
        combinations=[1, 2],
        days=[1, 2],
        cd_sectors=_cd_sectors(n_cds, sectors_per_cd),
        daily_capacity=_capacity(n_cds),
        projected_demand=_demand(sectors),
        current_assignment={(s, 1): 1 for s in sectors},
        max_churn=len(sectors),
        days_by_cycle={0: [1, 2]},
    )
    base.update(overrides)
    return base


def _cd_sectors(n_cds: int, per_cd: int) -> dict[int, list[int]]:
    return {cd: list(range((cd - 1) * per_cd + 1, cd * per_cd + 1)) for cd in range(1, n_cds + 1)}


def _capacity(n_cds: int) -> dict[tuple[int, int], float]:
    return {(cd, day): 1000.0 for cd in range(1, n_cds + 1) for day in (1, 2)}


def _demand(sectors: list[int]) -> dict[tuple[int, int, int], float]:
    return {
        **{(s, 1, 1): 10.0 for s in sectors},
        **{(s, 2, 2): 10.0 for s in sectors},
    }


def _all_in(sectors: range, combination: int) -> dict[int, int]:
    return {s: combination for s in sectors}


class ScriptedSolver:
    """Solver de subproblemas simulado: aplica as escolhas do roteiro, uma por chamada.

    Em cada chamada grava quais setores estavam fixos/binários/relaxados no modelo,
    depois atribui aos setores binários a combinação do roteiro. Sem roteiro sobrando,
    responde que não achou solução.
    """

    def __init__(self, script: list[dict[int, int] | None]) -> None:
        self.script = list(script)
        self.snapshots: list[dict[str, set[int]]] = []
        self.time_limits: list[float | None] = []

    def __call__(self, model, time_limit) -> bool:
        self.snapshots.append(_snapshot(model))
        self.time_limits.append(time_limit)
        choice = self.script.pop(0) if self.script else None  # None: sem solução nesta chamada
        if choice is None:
            return False
        for (sector, combination), var in model.x.items():
            _apply(var, sector, combination, choice)
        return True


def _apply(var, sector: int, combination: int, choice: dict[int, int]) -> None:
    """Escreve a escolha do roteiro nas variáveis livres: binárias valem 0/1, relaxadas 0."""
    if var.fixed:
        return
    chosen = var.is_binary() and choice[sector] == combination
    var.set_value(1 if chosen else 0)


def _mode(var) -> str:
    if var.fixed:
        return "fixed"
    return "binary" if var.is_binary() else "relaxed"


def _snapshot(model) -> dict[str, set[int]]:
    snapshot: dict[str, set[int]] = {"fixed": set(), "binary": set(), "relaxed": set()}
    for (sector, _), var in model.x.items():
        snapshot[_mode(var)].add(sector)
    return snapshot


def test_relax_and_fix_windows_set_variable_modes_per_iteration():
    solver = ScriptedSolver([{1: 2, 2: 1}, {3: 2, 4: 1}])
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=0)

    solve(**_instance(), params=params, solve_subproblem=solver)

    assert solver.snapshots == [
        {"fixed": set(), "binary": {1, 2}, "relaxed": {3, 4}},
        {"fixed": {1, 2}, "binary": {3, 4}, "relaxed": set()},
    ]


def test_overlapping_windows_reoptimize_the_cd_left_unfrozen():
    # 3 CDs, W = 2, P = 1: o CD 2 fica binário na 1a e na 2a iteração.
    solver = ScriptedSolver([{1: 2, 2: 1, 3: 2, 4: 1}, {3: 1, 4: 2, 5: 2, 6: 1}])
    params = RelaxAndFixParams(window=2, step=1, optimize_passes=0)

    result = solve(**_instance(3), params=params, solve_subproblem=solver)

    assert solver.snapshots == [
        {"fixed": set(), "binary": {1, 2, 3, 4}, "relaxed": {5, 6}},
        {"fixed": {1, 2}, "binary": {3, 4, 5, 6}, "relaxed": set()},
    ]
    # O CD 2 (setores 3, 4) ficou com a escolha da 2a iteração, não da 1a.
    assert result.assignment == {1: 2, 2: 1, 3: 1, 4: 2, 5: 2, 6: 1}


def test_construction_result_is_the_union_of_the_frozen_steps():
    solver = ScriptedSolver([{1: 2, 2: 2}, {3: 1, 4: 1}])
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=0)

    result = solve(**_instance(), params=params, solve_subproblem=solver)

    outcome = (result.assignment, result.objective, result.feasible)
    assert outcome == ({1: 2, 2: 2, 3: 1, 4: 1}, 0.0, True)
    assert (result.stop_reason, result.subproblems) == (COMPLETED, 2)


def test_fix_and_optimize_accepts_an_improving_window_and_keeps_the_rest_fixed():
    solver = ScriptedSolver(
        [
            {1: 2, 2: 1},  # R&F, CD 1
            {3: 1, 4: 1},  # R&F, CD 2 -> construção: dia 1 = 30, dia 2 = 10, amplitude 20
            {1: 2, 2: 2},  # F&O, libera o CD 1 -> dia 1 = 20, dia 2 = 20, amplitude 0
        ]
    )
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=1)

    result = solve(**_instance(), params=params, solve_subproblem=solver)

    assert result.construction.objective == 20.0
    assert result.objective == 0.0
    assert result.assignment == {1: 2, 2: 2, 3: 1, 4: 1}
    # No F&O do CD 1, os setores do CD 2 estavam fixos na incumbente e os do CD 1 binários.
    assert solver.snapshots[2] == {"fixed": {3, 4}, "binary": {1, 2}, "relaxed": set()}


def test_fix_and_optimize_never_returns_something_worse_than_the_construction():
    solver = ScriptedSolver(
        [
            {1: 2, 2: 2},  # R&F, CD 1
            {3: 1, 4: 1},  # R&F, CD 2 -> amplitude 0
            {1: 1, 2: 1},  # F&O tenta uma piora (amplitude 20): deve ser rejeitada
            {3: 2, 4: 2},  # F&O no CD 2, também pior
        ]
    )
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=1)

    result = solve(**_instance(), params=params, solve_subproblem=solver)

    assert result.assignment == {1: 2, 2: 2, 3: 1, 4: 1}
    assert result.objective == result.construction.objective == 0.0


def test_fix_and_optimize_rejects_a_candidate_that_breaks_the_churn_budget():
    solver = ScriptedSolver(
        [
            {1: 1, 2: 1},  # R&F, CD 1: sem mudança
            {3: 1, 4: 1},  # R&F, CD 2: sem mudança -> amplitude 40, churn 0
            {1: 2, 2: 2},  # F&O: amplitude cai pra 20, mas muda 2 setores (churn 4 > 2)
            {3: 1, 4: 1},
        ]
    )
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=1)

    result = solve(**_instance(max_churn=1), params=params, solve_subproblem=solver)

    assert result.assignment == {1: 1, 2: 1, 3: 1, 4: 1}
    assert result.objective == 40.0


def test_fix_and_optimize_stops_after_a_pass_without_improvement():
    # R&F (2 chamadas) + 1a varredura do F&O (2 janelas), sem melhora: a 2a varredura não roda.
    optimum = [{1: 2, 2: 1}, {3: 2, 4: 1}]
    solver = ScriptedSolver(optimum + optimum)
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=5)

    result = solve(**_instance(), params=params, solve_subproblem=solver)

    assert result.subproblems == 4
    assert result.objective == 0.0


def test_fix_and_optimize_can_be_disabled():
    solver = ScriptedSolver([{1: 2, 2: 1}, {3: 1, 4: 1}])
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=0)

    result = solve(**_instance(), params=params, solve_subproblem=solver)

    assert result.subproblems == 2
    assert result.construction == result.evaluation


def test_window_without_solution_returns_the_as_is_and_says_why():
    solver = ScriptedSolver([])  # o 1o subproblema já não acha solução
    params = RelaxAndFixParams(window=1, step=1)

    result = solve(**_instance(), params=params, solve_subproblem=solver)

    assert result.stop_reason == NO_SOLUTION_WINDOW
    assert result.assignment == {1: 1, 2: 1, 3: 1, 4: 1}
    assert result.construction is None
    assert result.subproblems == 1


def test_subproblem_time_limit_is_forwarded_to_every_solve():
    solver = ScriptedSolver([{1: 2, 2: 1}, {3: 2, 4: 1}])
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=0, subproblem_time_limit=7.0)

    solve(**_instance(), params=params, solve_subproblem=solver)

    assert solver.time_limits == [7.0, 7.0]


def test_cd_order_changes_which_cds_enter_the_first_window():
    solver = ScriptedSolver([{3: 2, 4: 1}, {1: 2, 2: 1}])
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=0)

    solve(**_instance(), params=params, solve_subproblem=solver, cd_order=[2, 1])

    assert solver.snapshots[0]["binary"] == {3, 4}


@pytest.mark.parametrize("cd_order", [[1], [1, 1], [1, 2, 3]])
def test_cd_order_must_list_every_cd_exactly_once(cd_order):
    with pytest.raises(ValueError, match="cd_order"):
        solve(**_instance(), solve_subproblem=ScriptedSolver([]), cd_order=cd_order)


def test_sector_without_cd_is_rejected_before_any_solve():
    solver = ScriptedSolver([])

    with pytest.raises(ValueError, match="sem CD"):
        solve(**_instance(cd_sectors={1: [1, 2], 2: [3]}), solve_subproblem=solver)

    assert solver.snapshots == []


def test_max_group_sectors_splits_each_cd_into_slices_that_form_the_windows():
    # 2 CDs com 4 setores cada, fatias de até 2: 4 grupos; janela de 1 grupo.
    solver = ScriptedSolver([_all_in(range(1, 9), 1)] * 4)
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=0, max_group_sectors=2)

    solve(**_instance(2, sectors_per_cd=4), params=params, solve_subproblem=solver)

    assert [s["binary"] for s in solver.snapshots] == [{1, 2}, {3, 4}, {5, 6}, {7, 8}]
    assert solver.snapshots[1] == {"fixed": {1, 2}, "binary": {3, 4}, "relaxed": {5, 6, 7, 8}}


def test_slices_of_one_cd_are_balanced():
    solver = ScriptedSolver([_all_in(range(1, 6), 1)] * 2)
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=0, max_group_sectors=3)

    solve(**_instance(1, sectors_per_cd=5), params=params, solve_subproblem=solver)

    # 5 setores em fatias de no máximo 3 -> 2 fatias de 2 e 3 setores (não 3 + 2 nem 3 + 1 + 1).
    assert [len(s["binary"]) for s in solver.snapshots] == [2, 3]


def test_cd_smaller_than_the_slice_size_stays_whole():
    solver = ScriptedSolver([{1: 2, 2: 1}, {3: 2, 4: 1}])
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=0, max_group_sectors=10)

    solve(**_instance(), params=params, solve_subproblem=solver)

    assert [s["binary"] for s in solver.snapshots] == [{1, 2}, {3, 4}]


@pytest.mark.parametrize("size", [0, -3])
def test_max_group_sectors_must_be_positive(size):
    with pytest.raises(ValueError, match="max_group_sectors"):
        RelaxAndFixParams(max_group_sectors=size)


def test_failed_window_is_retried_with_the_last_frozen_group_released():
    # 3 CDs, W = 1, P = 1: a 2a janela (CD 2) não acha solução; a 2a tentativa descongela o CD 1.
    solver = ScriptedSolver(
        [
            {1: 2, 2: 1},  # janela 1 (CD 1)
            None,  # janela 2 (CD 2): sem solução
            {1: 1, 2: 2, 3: 2, 4: 1},  # janela 2 alargada: CD 1 + CD 2
            {5: 2, 6: 1},  # janela 3 (CD 3), com CD 1 e 2 fixos no valor novo
        ]
    )
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=0, max_backtracks=1)

    result = solve(**_instance(3), params=params, solve_subproblem=solver)

    assert solver.snapshots[2:] == [
        {"fixed": set(), "binary": {1, 2, 3, 4}, "relaxed": {5, 6}},
        {"fixed": {1, 2, 3, 4}, "binary": {5, 6}, "relaxed": set()},
    ]
    assert result.assignment == {1: 1, 2: 2, 3: 2, 4: 1, 5: 2, 6: 1}
    assert (result.stop_reason, result.subproblems) == (COMPLETED, 4)


def test_backtracking_gives_up_after_max_backtracks_retries():
    solver = ScriptedSolver([{1: 2, 2: 1}, None, None])
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=0, max_backtracks=1)

    result = solve(**_instance(3), params=params, solve_subproblem=solver)

    assert result.stop_reason == NO_SOLUTION_WINDOW
    assert result.subproblems == 3


def test_backtracking_can_be_disabled():
    solver = ScriptedSolver([{1: 2, 2: 1}, None])
    params = RelaxAndFixParams(window=1, step=1, optimize_passes=0, max_backtracks=0)

    result = solve(**_instance(3), params=params, solve_subproblem=solver)

    assert result.stop_reason == NO_SOLUTION_WINDOW
    assert result.subproblems == 2


def test_first_window_has_nothing_to_release_so_it_is_not_retried():
    solver = ScriptedSolver([None, None, None])
    params = RelaxAndFixParams(window=1, step=1, max_backtracks=5)

    result = solve(**_instance(3), params=params, solve_subproblem=solver)

    assert result.stop_reason == NO_SOLUTION_WINDOW
    assert result.subproblems == 1


def test_max_backtracks_must_not_be_negative():
    with pytest.raises(ValueError, match="max_backtracks"):
        RelaxAndFixParams(max_backtracks=-1)


@patch("src.solver.matheuristics.relax_and_fix.solve")
def test_run_relax_and_fix_reports_a_solve_result_for_a_feasible_run(mock_solve):
    mock_solve.return_value = _fake_result(feasible=True)

    result = run_relax_and_fix(**_instance())

    assert isinstance(result, SolveResult)
    assert (result.model_name, result.solver, result.status) == (
        "relax_and_fix",
        "appsi_highs",
        COMPLETED,
    )
    assert result.termination_condition == "feasible"
    assert result.objective == 5.0


@patch("src.solver.matheuristics.relax_and_fix.solve")
def test_run_relax_and_fix_hides_the_objective_of_an_infeasible_run(mock_solve):
    mock_solve.return_value = _fake_result(feasible=False)

    result = run_relax_and_fix(**_instance())

    assert result.termination_condition == "infeasible"
    assert result.objective is None


def _fake_result(feasible: bool):
    from src.solver.matheuristics.evaluation import Evaluation
    from src.solver.matheuristics.relax_and_fix import RelaxAndFixResult

    evaluation = Evaluation(
        objective=5.0,
        capacity_violation=0.0 if feasible else 3.0,
        churn_budget_used=0.0,
        churn_budget=10.0,
    )
    return RelaxAndFixResult(
        assignment={},
        evaluation=evaluation,
        construction=evaluation,
        subproblems=3,
        stop_reason=COMPLETED,
        time=0.25,
    )
