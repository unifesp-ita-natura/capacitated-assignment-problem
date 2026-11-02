"""Tests for the compare_modeling experiment runner — the solver backend is mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from experiments.compare_modeling.run import run


@patch("experiments.compare_modeling.run.run_block_assignment_mip")
def test_run_produces_one_result_per_solver(mock_run_block_assignment_mip):
    def fake_run(instance, *, solver, **kwargs):
        result = MagicMock()
        result.solver = solver
        result.model_name = "mip_block_assignment"
        return result

    mock_run_block_assignment_mip.side_effect = fake_run

    results = run(solvers=["gurobi", "appsi_highs"], sector_sweep=[40], cycle_sweep=[1])

    assert [r.solver for r in results] == ["gurobi", "appsi_highs"]
    assert all(r.model_name == "mip_block_assignment" for r in results)
