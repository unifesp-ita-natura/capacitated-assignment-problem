"""Testes da tradução de tolerância e tempo para os nomes nativos de cada solver do MIP."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pyomo.environ as pyo
import pytest

from src.calibration.config import MipSettings
from src.calibration.mip import is_gurobi, solve_kwargs, solve_mip


@pytest.mark.parametrize("solver", ["gurobi", "appsi_gurobi", "gurobi_direct", "gurobi_persistent"])
def test_gurobi_gets_mipgap_and_timelimit_as_native_options(solver):
    assert solve_kwargs(solver, 1e-4, 600) == {"options": {"MIPGap": 1e-4, "TimeLimit": 600}}


def test_highs_keeps_mip_rel_gap_and_the_generic_timelimit():
    assert solve_kwargs("appsi_highs", 0.01, 600) == {
        "timelimit": 600,
        "options": {"mip_rel_gap": 0.01},
    }


def test_extra_options_are_merged():
    kwargs = solve_kwargs("appsi_gurobi", 0.01, 60, {"Threads": 2, "NodefileStart": 0.5})

    assert kwargs["options"] == {
        "MIPGap": 0.01,
        "TimeLimit": 60,
        "Threads": 2,
        "NodefileStart": 0.5,
    }


def test_is_gurobi():
    assert is_gurobi("appsi_gurobi") and not is_gurobi("appsi_highs")


def _closed_solver() -> MagicMock:
    solver = MagicMock()
    solver.solve.return_value = SimpleNamespace(
        problem=SimpleNamespace(lower_bound=0.0, upper_bound=0.0),
        solver=SimpleNamespace(termination_condition="optimal"),
    )
    return solver


def test_solve_mip_passes_native_gurobi_options_and_records_them(tiny_instance):
    solver = _closed_solver()
    with patch("src.calibration.mip.pyo.SolverFactory", return_value=solver):
        outcome = solve_mip(tiny_instance, MipSettings(solver="appsi_gurobi", mip_rel_gap=1e-4), 30)

    kwargs = solver.solve.call_args.kwargs
    assert kwargs["options"] == {"MIPGap": 1e-4, "TimeLimit": 30}
    assert "timelimit" not in kwargs
    assert outcome.options == {"MIPGap": 1e-4, "TimeLimit": 30}
    assert outcome.mip_rel_gap == 1e-4


def test_highs_closes_the_two_sector_instance_for_real(tiny_instance):
    outcome = solve_mip(tiny_instance, MipSettings(solver="appsi_highs", mip_rel_gap=1e-4), 30)

    assert outcome.closed
    assert outcome.objective == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(
    not pyo.SolverFactory("appsi_gurobi").available(exception_flag=False),
    reason="Gurobi sem licença/instalação nesta máquina",
)
def test_gurobi_closes_the_two_sector_instance_for_real(tiny_instance):
    outcome = solve_mip(tiny_instance, MipSettings(solver="appsi_gurobi", mip_rel_gap=1e-4), 30)

    assert outcome.closed
    assert outcome.objective == pytest.approx(0.0, abs=1e-9)
