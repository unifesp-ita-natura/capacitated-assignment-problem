"""Tests for the compare_modeling experiment runner — the solver backend is mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from experiments.compare_modeling.run import run


@patch("experiments.compare_modeling.run.solve_example")
def test_run_produces_one_result_per_solver(mock_solve_example):
    mock_results = MagicMock()
    mock_results.solver.status = "ok"
    mock_results.solver.termination_condition = "optimal"
    mock_solve_example.return_value = mock_results

    results = run(solvers=["gurobi", "appsi_highs"])

    assert [r.solver for r in results] == ["gurobi", "appsi_highs"]
    assert all(r.model_name == "example" for r in results)
