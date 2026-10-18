"""Tests for the sector/Bloco-Subloco assignment MIP.

Solves run against the real HiGHS backend (bundled via `highspy`, no license
needed) rather than a mocked solver, so these tests exercise real solving.
A fake "gurobi" solver is used only to force the one condition a license-free
machine can't produce for real: Gurobi being unavailable.
"""

from __future__ import annotations

import pytest

from src.solver.mip.block_assignment import (
    build_block_assignment_model,
    run_block_assignment_mip,
    solve_block_assignment,
)


class _UnavailableGurobiStub:
    """Stands in for `pyo.SolverFactory("gurobi")` when no license is installed."""

    def available(self, exception_flag: bool = True) -> bool:
        return False

    def solve(self, *args, **kwargs):
        raise AssertionError("gurobi stub should never be solved against")


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
        days_by_cycle={0: [1, 2]},
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


def test_z_max_and_z_min_are_indexed_by_horizon_cycle(toy_instance):
    model = build_block_assignment_model(**toy_instance)

    assert set(model.H) == set(toy_instance["days_by_cycle"])
    assert len(model.z_max) == len(toy_instance["days_by_cycle"])
    assert len(model.z_min) == len(toy_instance["days_by_cycle"])


def test_demand_range_constraints_scope_each_cycle_to_its_own_days():
    instance = dict(
        sectors=[1, 2],
        combinations=[1, 2],
        days=[1, 2, 11, 12],
        cd_sectors={1: [1, 2]},
        daily_capacity={(1, 1): 100.0, (1, 2): 100.0, (1, 11): 100.0, (1, 12): 100.0},
        projected_demand={
            (1, 1, 1): 10.0,
            (1, 2, 1): 0.0,
            (1, 11, 1): 10.0,
            (1, 12, 1): 0.0,
            (2, 1, 1): 10.0,
            (2, 2, 1): 0.0,
            (2, 11, 1): 10.0,
            (2, 12, 1): 0.0,
        },
        current_assignment={(1, 1): 1, (2, 1): 1},
        max_churn=2,
        days_by_cycle={0: [1, 2], 1: [11, 12]},
    )

    model = build_block_assignment_model(**instance)

    assert set(model.demand_upper_bound) == {(0, 1), (0, 2), (1, 11), (1, 12)}


def test_capacity_constraint_covers_every_cd_and_day(toy_instance):
    model = build_block_assignment_model(**toy_instance)

    expected = len(toy_instance["cd_sectors"]) * len(toy_instance["days"])
    assert len(model.cd_capacity) == expected


def test_churn_constraint_is_a_single_budget_across_all_sectors(toy_instance):
    model = build_block_assignment_model(**toy_instance)

    assert model.churn_limit.active
    assert model.churn_limit.upper() == 2 * toy_instance["max_churn"]


def test_solve_block_assignment_solves_for_real_with_highs(toy_instance):
    model = build_block_assignment_model(**toy_instance)

    results = solve_block_assignment(model, solver="appsi_highs")

    assert str(results.solver.termination_condition) == "optimal"


def test_run_block_assignment_mip_defaults_to_gurobi_when_a_license_is_installed(toy_instance):
    result = run_block_assignment_mip(**toy_instance)

    assert result.model_name == "mip_block_assignment"
    assert result.solver in ("gurobi", "appsi_highs")  # falls back if this machine has no license
    assert result.termination_condition == "optimal"
    assert result.wall_time_seconds >= 0


def test_run_block_assignment_mip_dispatches_to_requested_solver(toy_instance):
    result = run_block_assignment_mip(**toy_instance, solver="appsi_highs")

    assert result.solver == "appsi_highs"
    assert result.termination_condition == "optimal"


def test_run_block_assignment_mip_reports_no_solution_instead_of_raising():
    """Capacity far below demand makes this infeasible, so HiGHS/Gurobi find no
    feasible solution to load — `run_block_assignment_mip` should report that
    via `SolveResult`, not propagate Pyomo's `RuntimeError`.
    """
    infeasible_instance = dict(
        sectors=[1, 2],
        combinations=[1, 2],
        days=[1, 2],
        cd_sectors={1: [1, 2]},
        daily_capacity={(1, 1): 1.0, (1, 2): 1.0},
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
        days_by_cycle={0: [1, 2]},
    )

    result = run_block_assignment_mip(**infeasible_instance, solver="appsi_highs")

    assert result.status == "aborted"
    assert result.termination_condition == "noSolution"
    assert result.objective is None


def test_solve_block_assignment_falls_back_to_highs_when_gurobi_unavailable(
    monkeypatch, toy_instance, caplog
):
    import pyomo.environ as pyo

    real_solver_factory = pyo.SolverFactory
    monkeypatch.setattr(
        "src.solver.mip.block_assignment.pyo.SolverFactory",
        lambda name: _UnavailableGurobiStub() if name == "gurobi" else real_solver_factory(name),
    )
    model = build_block_assignment_model(**toy_instance)

    with caplog.at_level("WARNING"):
        results = solve_block_assignment(model)

    assert str(results.solver.termination_condition) == "optimal"
    assert "falling back" in caplog.text.lower()


def test_run_block_assignment_mip_falls_back_to_highs_when_gurobi_unavailable(
    monkeypatch, toy_instance
):
    import pyomo.environ as pyo

    real_solver_factory = pyo.SolverFactory
    monkeypatch.setattr(
        "src.solver.mip.block_assignment.pyo.SolverFactory",
        lambda name: _UnavailableGurobiStub() if name == "gurobi" else real_solver_factory(name),
    )

    result = run_block_assignment_mip(**toy_instance)

    assert result.solver == "appsi_highs"
    assert result.termination_condition == "optimal"
