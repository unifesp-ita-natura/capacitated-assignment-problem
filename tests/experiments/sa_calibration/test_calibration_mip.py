"""Testes do MIP de referência da calibração — o solver é sempre mockado, nunca chamado."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.calibration.config import MipSettings
from src.calibration.mip import (
    assignment_from_model,
    finite_or_none,
    relative_gap,
    solve_mip,
)


def _fake_results(lower: float, upper: float, termination: str = "optimal"):
    return SimpleNamespace(
        problem=SimpleNamespace(lower_bound=lower, upper_bound=upper),
        solver=SimpleNamespace(termination_condition=termination),
    )


def _fake_solver(results, chosen: dict[tuple[int, int], float] | None = None) -> MagicMock:
    solver = MagicMock()
    solver.solve.return_value = results

    def load_vars():
        model = solver.solve.call_args.args[0]
        for index, var in model.x.items():
            var.set_value((chosen or {}).get(index, 0.0))

    solver.load_vars.side_effect = load_vars
    return solver


@pytest.mark.parametrize(
    ("value", "expected"), [(1.5, 1.5), (math.inf, None), (-math.inf, None), (None, None)]
)
def test_finite_or_none(value, expected):
    assert finite_or_none(value) == expected


def test_relative_gap():
    assert relative_gap(90.0, 100.0) == pytest.approx(0.1)
    assert relative_gap(None, 100.0) is None
    assert relative_gap(0.0, 0.0) is None


def _solve_closed(tiny_instance):
    solver = _fake_solver(_fake_results(0.0, 0.0), chosen={(1, 1): 1.0, (2, 2): 1.0})
    with patch("src.calibration.mip.pyo.SolverFactory", return_value=solver) as factory:
        outcome = solve_mip(tiny_instance, MipSettings(mip_rel_gap=0.01), time_limit=600)
    return factory, solver, outcome


def test_solve_mip_uses_highs_with_time_limit_and_gap(tiny_instance):
    factory, solver, _ = _solve_closed(tiny_instance)

    factory.assert_called_once_with("appsi_highs")
    kwargs = solver.solve.call_args.kwargs
    assert (kwargs["timelimit"], kwargs["options"]) == (600, {"mip_rel_gap": 0.01})
    assert kwargs["load_solutions"] is False


def test_solve_mip_reads_the_incumbent_assignment(tiny_instance):
    _, _, outcome = _solve_closed(tiny_instance)

    assert outcome.closed
    assert outcome.has_incumbent
    assert outcome.x == {1: 1, 2: 2}
    assert outcome.to_dict()["x"] == {"1": 1, "2": 2}


def test_solve_mip_without_incumbent_loads_nothing(tiny_instance):
    solver = _fake_solver(_fake_results(-math.inf, math.inf, "maxTimeLimit"))
    with patch("src.calibration.mip.pyo.SolverFactory", return_value=solver):
        outcome = solve_mip(tiny_instance, MipSettings(), time_limit=1)

    solver.load_vars.assert_not_called()
    assert (outcome.closed, outcome.has_incumbent, outcome.x) == (False, False, {})
    assert outcome.lower_bound is None and outcome.gap is None


def test_assignment_from_model_reads_the_chosen_combinations():
    variables = {(1, 1): SimpleNamespace(value=1.0), (1, 2): SimpleNamespace(value=None)}
    model = SimpleNamespace(x=variables)

    assert assignment_from_model(model) == {1: 1}
