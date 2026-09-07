"""Tests for the sector/Bloco-Subloco assignment MIP — solver backend is mocked, never called."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.solver.mip.block_assignment import build_block_assignment_model, solve_block_assignment


@pytest.fixture
def toy_instance() -> dict:
    return dict(
        sectors=[1, 2],
        combinations=[1, 2],
        days=[1, 2],
        cd_sectors={1: [1, 2]},
        daily_capacity={(1, 1): 100.0, (1, 2): 100.0},
        projected_demand={
            (1, 1, 1): 10.0,
            (1, 2, 1): 0.0,
            (1, 1, 2): 0.0,
            (1, 2, 2): 10.0,
            (2, 1, 1): 10.0,
            (2, 2, 1): 0.0,
            (2, 1, 2): 0.0,
            (2, 2, 2): 10.0,
        },
        current_assignment={(1, 1): 1, (2, 2): 1},
        max_churn=1,
    )


def test_build_block_assignment_model_has_expected_structure(toy_instance):
    model = build_block_assignment_model(**toy_instance)

    assert hasattr(model, "x")
    assert hasattr(model, "z_max")
    assert hasattr(model, "z_min")
    assert hasattr(model, "objective")


def test_single_assignment_constraint_covers_every_sector(toy_instance):
    model = build_block_assignment_model(**toy_instance)

    assert len(model.single_assignment) == len(toy_instance["sectors"])


def test_demand_range_constraints_cover_every_day(toy_instance):
    model = build_block_assignment_model(**toy_instance)

    assert len(model.demand_upper_bound) == len(toy_instance["days"])
    assert len(model.demand_lower_bound) == len(toy_instance["days"])


def test_capacity_constraint_covers_every_cd_and_day(toy_instance):
    model = build_block_assignment_model(**toy_instance)

    expected = len(toy_instance["cd_sectors"]) * len(toy_instance["days"])
    assert len(model.cd_capacity) == expected


def test_churn_constraint_is_a_single_budget_across_all_sectors(toy_instance):
    model = build_block_assignment_model(**toy_instance)

    assert model.churn_limit.active
    assert model.churn_limit.upper() == 2 * toy_instance["max_churn"]


@patch("src.solver.mip.block_assignment.pyo.SolverFactory")
def test_solve_block_assignment_defaults_to_gurobi(mock_solver_factory, toy_instance):
    mock_solver = MagicMock()
    mock_solver_factory.return_value = mock_solver
    model = build_block_assignment_model(**toy_instance)

    solve_block_assignment(model)

    mock_solver_factory.assert_called_once_with("gurobi")
    mock_solver.solve.assert_called_once()


@pytest.mark.parametrize("solver_name", ["gurobi", "appsi_highs", "cbc"])
@patch("src.solver.mip.block_assignment.pyo.SolverFactory")
def test_solve_block_assignment_dispatches_to_requested_solver(
    mock_solver_factory, solver_name, toy_instance
):
    mock_solver = MagicMock()
    mock_solver_factory.return_value = mock_solver
    model = build_block_assignment_model(**toy_instance)

    solve_block_assignment(model, solver=solver_name)

    mock_solver_factory.assert_called_once_with(solver_name)
