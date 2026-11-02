"""Tests the forecast, solve and explain stages of the end-to-end pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.cli.main import TOY_INSTANCE
from src.config.schema import PipelineConfig, RunConfig
from src.forecasting.dataset import cycle_calendar, load_demand_base
from src.pipeline import explain
from src.pipeline.forecast_stage import pick_targets, plan_horizon
from src.pipeline.solve_stage import SOLVERS, run_solver

FIXTURE = "tests/fixtures/demand_sample.csv"


def _config(family: str) -> RunConfig:
    return RunConfig(solver="appsi_highs", pipeline=PipelineConfig(solver_family=family))


def test_pick_targets_takes_last_complete_cycles() -> None:
    calendar = cycle_calendar(load_demand_base(FIXTURE))
    assert pick_targets(calendar, 2, 8) == ["202612", "202613"]


def test_pick_targets_rejects_too_little_history() -> None:
    calendar = cycle_calendar(load_demand_base(FIXTURE))
    with pytest.raises(ValueError, match="need"):
        pick_targets(calendar, 2, 12)


def test_plan_horizon_cuts_off_before_first_target() -> None:
    plan = plan_horizon(load_demand_base(FIXTURE), 2, 8)
    assert plan.as_of == pd.Timestamp("2026-07-08")
    assert set(plan.windows.loc[plan.windows["emit"], "CICLOS"]) == set(plan.targets)


@pytest.mark.parametrize("family", sorted(SOLVERS))
def test_every_solver_returns_a_feasible_assignment(family: str) -> None:
    solved = run_solver(family, TOY_INSTANCE, _config(family))
    assert solved.assignment is not None
    assert set(solved.assignment) == set(TOY_INSTANCE.sectors)
    assert solved.result.objective is not None


def test_explanation_never_worsens_the_as_is_metric() -> None:
    solved = run_solver("mip", TOY_INSTANCE, _config("mip"))
    explanation = explain.explain(TOY_INSTANCE, solved, {"wmape": 0.1})
    assert explanation.to_be.objective <= explanation.as_is.objective + 1e-6
    assert "As-Is" in explain.render(explanation)
    frame = explain.to_frame(explanation)
    assert {"as_is", "to_be", "comparison", "forecast"} <= set(frame["scope"])


def test_explanation_reports_infeasible_solve() -> None:
    solved = run_solver("sa", TOY_INSTANCE, _config("sa"))._replace(assignment=None)
    explanation = explain.explain(TOY_INSTANCE, solved, {})
    assert explanation.to_be is None
    assert "no feasible" in explain.render(explanation)
