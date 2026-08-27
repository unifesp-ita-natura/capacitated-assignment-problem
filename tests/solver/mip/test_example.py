"""Tests for the example solver wiring — the solver backend is mocked, never called directly."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.solver.mip.example import build_example_model, solve_example


def test_build_example_model_has_expected_structure():
    model = build_example_model()

    assert hasattr(model, "x")
    assert hasattr(model, "y")
    assert hasattr(model, "capacity")
    assert hasattr(model, "objective")


@patch("src.solver.mip.example.pyo.SolverFactory")
def test_solve_example_defaults_to_gurobi(mock_solver_factory):
    mock_solver = MagicMock()
    mock_solver_factory.return_value = mock_solver

    solve_example()

    mock_solver_factory.assert_called_once_with("gurobi")
    mock_solver.solve.assert_called_once()


@pytest.mark.parametrize("solver_name", ["gurobi", "appsi_highs", "cbc"])
@patch("src.solver.mip.example.pyo.SolverFactory")
def test_solve_example_dispatches_to_requested_solver(mock_solver_factory, solver_name):
    mock_solver = MagicMock()
    mock_solver_factory.return_value = mock_solver

    solve_example(solver=solver_name)

    mock_solver_factory.assert_called_once_with(solver_name)
