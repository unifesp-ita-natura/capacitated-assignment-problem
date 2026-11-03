"""Score SA assignments on realized demand and summarize their spread across seeds and scenarios."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from src.pipeline.explain import describe
from src.pipeline.realized import rescore
from src.solver.heuristics import simulated_annealing
from src.solver.inputs import ProblemInstance
from src.solver.matheuristics.problem import build_problem


@dataclass(frozen=True)
class SeedOutcome:
    """One SA run: its objective on the forecast and on the realized demand, As-Is included."""

    seed: int
    forecast_as_is: float
    forecast_to_be: float
    realized_as_is: float
    realized_to_be: float
    sectors_moved: int
    seconds: float

    @property
    def forecast_improvement_pct(self) -> float:
        """Drop of the balance metric on the forecast demand, in percent of As-Is."""
        return _drop_pct(self.forecast_as_is, self.forecast_to_be)

    @property
    def realized_improvement_pct(self) -> float:
        """Drop of the balance metric on the realized demand, in percent of As-Is."""
        return _drop_pct(self.realized_as_is, self.realized_to_be)


def _drop_pct(before: float, after: float) -> float:
    return 100.0 * (before - after) / before if before else 0.0


def run_seed(seed: int, forecast: ProblemInstance, realized: ProblemInstance) -> SeedOutcome:
    """Solve the forecast instance with `seed`, then score As-Is and To-Be on both demands."""
    found = simulated_annealing.solve(forecast, rng=random.Random(seed))
    problem = build_problem(forecast)
    on_forecast = (
        describe(problem, forecast, forecast.current_assignment),
        describe(problem, forecast, found.assignment),
    )
    on_realized = rescore(forecast, realized, found.assignment)
    return SeedOutcome(
        seed=seed,
        forecast_as_is=on_forecast[0].objective,
        forecast_to_be=on_forecast[1].objective,
        realized_as_is=on_realized[0].objective,
        realized_to_be=on_realized[1].objective,
        sectors_moved=on_forecast[1].sectors_moved,
        seconds=found.time,
    )


def spread(values: pd.Series) -> dict[str, float]:
    """Mean, standard deviation, min, max and coefficient of variation of `values`."""
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
        "cv_pct": 100.0 * std / abs(mean) if mean else 0.0,
    }
