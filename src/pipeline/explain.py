"""Explain a solve: the balance metric As-Is vs To-Be, its constraints, and the forecast quality."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.pipeline.solve_stage import Solved
from src.solver.inputs import ProblemInstance
from src.solver.matheuristics.evaluation import capacity_utilization, evaluate_assignment
from src.solver.matheuristics.problem import AssignmentProblem, build_problem

METRIC_EXPLANATION = (
    "Balance metric = for each forecast cycle, the busiest day's total items minus the "
    "quietest day's, averaged over cycles. Days in the horizon's first and last 14 days, "
    "when not every slot is open, are left out. Lower means demand is spread more evenly "
    "across days, so the CDs see fewer peaks."
)


@dataclass(frozen=True)
class Scenario:
    """Metrics of one assignment (As-Is or To-Be) on the forecast demand."""

    objective: float
    mean_daily_items: float
    max_capacity_utilization: float
    capacity_violation: float
    sectors_moved: int

    @property
    def relative_imbalance(self) -> float:
        """Peak-to-trough range as a share of the mean daily load."""
        return self.objective / self.mean_daily_items if self.mean_daily_items else 0.0


@dataclass(frozen=True)
class Explanation:
    """As-Is vs To-Be comparison plus the solver's and forecast's own diagnostics."""

    as_is: Scenario
    to_be: Scenario | None
    churn_budget: float
    solver: str
    status: str
    wall_time_seconds: float
    forecast: dict[str, float]

    @property
    def improvement_pct(self) -> float | None:
        """Relative drop of the balance metric from As-Is to To-Be, in percent."""
        if self.to_be is None or not self.as_is.objective:
            return None
        return 100.0 * (self.as_is.objective - self.to_be.objective) / self.as_is.objective


def _daily_totals(instance: ProblemInstance, assignment: dict[int, int]) -> dict[int, float]:
    """Total items per day when every sector follows `assignment`."""
    daily: dict[int, float] = defaultdict(float)
    for (sector, day, combo), items in instance.projected_demand.items():
        if assignment.get(sector) == combo:
            daily[day] += items
    return daily


def _mean_daily_items(instance: ProblemInstance, assignment: dict[int, int]) -> float:
    """Average over cycles of the cycle's mean daily total items."""
    daily = _daily_totals(instance, assignment)
    means = [sum(daily[d] for d in days) / len(days) for days in instance.days_by_cycle.values()]
    return sum(means) / len(means) if means else 0.0


def describe(
    problem: AssignmentProblem, instance: ProblemInstance, assignment: dict[int, int]
) -> Scenario:
    """Evaluate one assignment with the same definition the MIP optimizes."""
    evaluation = evaluate_assignment(problem, assignment)
    return Scenario(
        objective=evaluation.objective,
        mean_daily_items=_mean_daily_items(instance, assignment),
        max_capacity_utilization=capacity_utilization(problem, assignment),
        capacity_violation=evaluation.capacity_violation,
        sectors_moved=sum(assignment[s] != c for s, c in instance.current_assignment.items()),
    )


def explain(instance: ProblemInstance, solved: Solved, forecast: dict[str, float]) -> Explanation:
    """Compare the As-Is calendar with the solver's assignment under the forecast demand."""
    problem = build_problem(instance)
    to_be = None
    if solved.assignment is not None:
        to_be = describe(problem, instance, solved.assignment)
    return Explanation(
        as_is=describe(problem, instance, instance.current_assignment),
        to_be=to_be,
        churn_budget=instance.max_churn,
        solver=solved.result.model_name,
        status=solved.result.termination_condition,
        wall_time_seconds=solved.result.wall_time_seconds,
        forecast=forecast,
    )


def _scenario_line(label: str, scenario: Scenario) -> str:
    return (
        f"  {label}: metric={scenario.objective:,.0f} items "
        f"({100 * scenario.relative_imbalance:.1f}% of mean daily load "
        f"{scenario.mean_daily_items:,.0f}), peak CD utilization="
        f"{100 * scenario.max_capacity_utilization:.1f}%, "
        f"capacity violation={scenario.capacity_violation:,.0f}"
    )


def _verdict_lines(explanation: Explanation) -> list[str]:
    if explanation.to_be is None:
        return [f"  {explanation.solver} found no feasible assignment ({explanation.status})."]
    improvement = explanation.improvement_pct
    gain = "n/a" if improvement is None else f"{improvement:.1f}%"
    return [
        _scenario_line("To-Be", explanation.to_be),
        f"  Improvement over As-Is: {gain}; sectors moved: {explanation.to_be.sectors_moved} "
        f"(budget {explanation.churn_budget:.0f}).",
    ]


def _forecast_lines(forecast: dict[str, float]) -> list[str]:
    if not forecast:
        return []
    shown = ", ".join(f"{name}={value:.3f}" for name, value in forecast.items())
    return [f"Forecast accuracy on the held-out horizon: {shown}."]


def render(explanation: Explanation) -> str:
    """The human-readable report printed at the end of the pipeline."""
    lines = [
        METRIC_EXPLANATION,
        _scenario_line("As-Is", explanation.as_is),
        *_verdict_lines(explanation),
        f"Solver: {explanation.solver} ({explanation.status}) in "
        f"{explanation.wall_time_seconds:.1f}s.",
        *_forecast_lines(explanation.forecast),
    ]
    return "\n".join(lines)


def to_frame(explanation: Explanation) -> pd.DataFrame:
    """One row per metric (As-Is, To-Be, improvement, forecast) for the explanation CSV."""
    rows = [("as_is", name, value) for name, value in vars(explanation.as_is).items()]
    if explanation.to_be is not None:
        rows += [("to_be", name, value) for name, value in vars(explanation.to_be).items()]
        rows.append(("comparison", "improvement_pct", explanation.improvement_pct))
    rows += [("forecast", name, value) for name, value in explanation.forecast.items()]
    return pd.DataFrame(rows, columns=["scope", "metric", "value"])


def write_explanation(explanation: Explanation, path: str | Path) -> None:
    """Persist `to_frame(explanation)` as CSV, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_frame(explanation).to_csv(path, index=False)
