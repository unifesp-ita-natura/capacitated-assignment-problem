"""Testes da preparação dos subproblemas e do solver Pyomo — o backend é mockado, nunca chamado."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pyomo.environ as pyo
import pytest

from src.solver.inputs import ProblemInstance
from src.solver.matheuristics.subproblem import (
    apply_variable_modes,
    mip_gap_option,
    pyomo_subproblem_solver,
    read_assignment,
)
from src.solver.mip.block_assignment import build_block_assignment_model


@pytest.fixture
def model() -> pyo.ConcreteModel:
    return build_block_assignment_model(
        ProblemInstance(
            sectors=[1, 2, 3],
            combinations=[1, 2],
            days=[1, 2],
            cd_sectors={1: [1, 2], 2: [3]},
            daily_capacity={(1, 1): 100.0, (1, 2): 100.0, (2, 1): 100.0, (2, 2): 100.0},
            projected_demand={(s, 1, 1): 10.0 for s in (1, 2, 3)},
            current_assignment={s: 1 for s in (1, 2, 3)},
            max_churn=3,
            days_by_cycle={0: [1, 2]},
        )
    )


def _modes(model: pyo.ConcreteModel, sector: int) -> set[str]:
    out = set()
    for (s, _), var in model.x.items():
        if s == sector:
            out.add("fixed" if var.fixed else "binary" if var.is_binary() else "relaxed")
    return out


def test_apply_variable_modes_fixes_frees_and_relaxes_by_sector(model):
    apply_variable_modes(model, fixed={1: 2}, binary={2})

    assert _modes(model, 1) == {"fixed"}
    assert _modes(model, 2) == {"binary"}
    assert _modes(model, 3) == {"relaxed"}


def test_fixed_sector_gets_one_on_its_combination_and_zero_elsewhere(model):
    apply_variable_modes(model, fixed={1: 2}, binary=set())

    assert (model.x[1, 1].value, model.x[1, 2].value) == (0, 1)


def test_modes_can_be_reapplied_to_change_a_sector_back_and_forth(model):
    apply_variable_modes(model, fixed={1: 2}, binary={2})
    apply_variable_modes(model, fixed={2: 1}, binary={1, 3})

    assert _modes(model, 1) == {"binary"}
    assert _modes(model, 2) == {"fixed"}
    assert _modes(model, 3) == {"binary"}


def test_relaxed_variables_accept_fractional_values(model):
    apply_variable_modes(model, fixed={}, binary=set())

    model.x[1, 1].set_value(0.4)  # não levanta: o domínio é [0, 1]
    assert model.x[1, 1].value == 0.4


def test_read_assignment_takes_the_combination_with_the_largest_x(model):
    for (sector, combination), var in model.x.items():
        var.set_value(1.0 if combination == 2 and sector != 3 else 0.0)
    model.x[3, 1].set_value(0.7)
    model.x[3, 2].set_value(0.3)

    assert read_assignment(model, [1, 2, 3]) == {1: 2, 2: 2, 3: 1}


def test_read_assignment_treats_unloaded_values_as_zero(model):
    assert set(read_assignment(model, [1])) == {1}


def _mock_solver(mock_factory: MagicMock, upper_bound) -> MagicMock:
    solver = MagicMock()
    solver.solve.return_value = MagicMock(problem=MagicMock(upper_bound=upper_bound))
    mock_factory.return_value = solver
    return solver


@patch("src.solver.matheuristics.subproblem.pyo.SolverFactory")
def test_solver_loads_the_solution_and_reports_success(mock_factory, model):
    solver = _mock_solver(mock_factory, upper_bound=12.5)

    solved = pyomo_subproblem_solver("appsi_highs")(model, 30.0)

    assert solved is True
    mock_factory.assert_called_once_with("appsi_highs")
    solver.load_vars.assert_called_once()


@patch("src.solver.matheuristics.subproblem.pyo.SolverFactory")
def test_solver_passes_time_limit_gap_and_defers_loading(mock_factory, model):
    solver = _mock_solver(mock_factory, upper_bound=1.0)

    pyomo_subproblem_solver(mip_rel_gap=1e-3)(model, 30.0)

    _, kwargs = solver.solve.call_args
    assert kwargs["timelimit"] == 30.0
    assert kwargs["options"] == {"mip_rel_gap": 1e-3}
    assert kwargs["load_solutions"] is False


@patch("src.solver.matheuristics.subproblem.pyo.SolverFactory")
def test_solver_omits_the_time_limit_when_none(mock_factory, model):
    solver = _mock_solver(mock_factory, upper_bound=1.0)

    pyomo_subproblem_solver()(model, None)

    assert "timelimit" not in solver.solve.call_args.kwargs


@patch("src.solver.matheuristics.subproblem.pyo.SolverFactory")
def test_solver_is_created_once_and_reused_across_subproblems(mock_factory, model):
    solver = _mock_solver(mock_factory, upper_bound=1.0)
    subproblem_solver = pyomo_subproblem_solver()

    subproblem_solver(model, 10.0)
    subproblem_solver(model, 10.0)
    subproblem_solver(model, 10.0)

    mock_factory.assert_called_once_with("appsi_highs")
    assert solver.solve.call_count == 3


@patch("src.solver.matheuristics.subproblem.pyo.SolverFactory")
def test_solver_is_not_created_until_the_first_solve(mock_factory):
    pyomo_subproblem_solver()

    mock_factory.assert_not_called()


@patch("src.solver.matheuristics.subproblem.pyo.SolverFactory")
def test_fixed_variables_are_treated_as_variables_for_cheap_updates(mock_factory, model):
    solver = _mock_solver(mock_factory, upper_bound=1.0)

    pyomo_subproblem_solver()(model, 10.0)

    assert solver.update_config.treat_fixed_vars_as_params is False


@patch("src.solver.matheuristics.subproblem.pyo.SolverFactory")
def test_solver_without_an_update_config_is_used_as_is(mock_factory, model):
    solver = _mock_solver(mock_factory, upper_bound=1.0)
    del solver.update_config  # um solver não persistente não tem esse atributo

    assert pyomo_subproblem_solver("cbc")(model, 10.0) is True


@pytest.mark.parametrize("upper_bound", [None, float("inf")])
@patch("src.solver.matheuristics.subproblem.pyo.SolverFactory")
def test_solver_reports_failure_without_an_incumbent(mock_factory, upper_bound, model):
    solver = _mock_solver(mock_factory, upper_bound=upper_bound)

    solved = pyomo_subproblem_solver()(model, 30.0)

    assert solved is False
    solver.load_vars.assert_not_called()


@patch("src.solver.matheuristics.subproblem.pyo.SolverFactory")
def test_solver_logs_the_termination_condition_when_there_is_no_solution(
    mock_factory, model, caplog
):
    solver = _mock_solver(mock_factory, upper_bound=None)
    solver.solve.return_value.solver.termination_condition = "maxTimeLimit"

    pyomo_subproblem_solver()(model, 5.0)

    assert "maxTimeLimit" in caplog.text


@pytest.mark.parametrize(
    ("solver_name", "option"),
    [("appsi_highs", "mip_rel_gap"), ("appsi_gurobi", "MIPGap"), ("cbc", "mip_rel_gap")],
)
def test_gap_option_name_follows_the_solver(solver_name, option):
    assert mip_gap_option(solver_name) == option


@patch("src.solver.matheuristics.subproblem.pyo.SolverFactory")
def test_gurobi_receives_its_own_gap_option_name(mock_factory, model):
    solver = _mock_solver(mock_factory, upper_bound=1.0)

    pyomo_subproblem_solver("appsi_gurobi", mip_rel_gap=1e-3)(model, 30.0)

    assert solver.solve.call_args.kwargs["options"] == {"MIPGap": 1e-3}
    mock_factory.assert_called_once_with("appsi_gurobi")
