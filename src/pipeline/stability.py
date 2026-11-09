"""Score SA assignments on realized demand and summarize their spread across seeds and scenarios."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
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
    iterations: int = 0
    stop_reason: str = ""

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
        iterations=found.iterations,
        stop_reason=found.stop_reason,
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


def selection_cell(frame: pd.DataFrame) -> dict[str, float]:
    """Does the seed with the lowest predicted objective also realize the best improvement?

    `frame` holds the seeds of one (scenario, window) cell. Returns the Spearman correlation
    between predicted and realized objective, and the realized improvement (percent of As-Is) of
    the picked seed, of the average seed and of the oracle (best realized) seed.
    """
    picked = frame.loc[frame["forecast_to_be"].idxmin()]
    return {
        "spearman": float(frame["forecast_to_be"].corr(frame["realized_to_be"], method="spearman")),
        "picked_improvement_pct": float(picked["realized_improvement_pct"]),
        "mean_improvement_pct": float(frame["realized_improvement_pct"].mean()),
        "oracle_improvement_pct": float(frame["realized_improvement_pct"].max()),
    }


def best_of_k(frame: pd.DataFrame, ks: list[int], draws: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Expected realized improvement when keeping the lowest-predicted seed of k random seeds.

    Bootstraps `draws` subsets of k seeds without replacement for each k <= the number of seeds.
    """
    rng = np.random.default_rng(seed)
    forecast = frame["forecast_to_be"].to_numpy()
    realized = frame["realized_improvement_pct"].to_numpy()
    rows = [
        {"k": k, "mean_improvement_pct": _mean_picked(forecast, realized, k, draws, rng)}
        for k in ks
        if k <= len(frame)
    ]
    return pd.DataFrame(rows)


def _mean_picked(
    forecast: np.ndarray, realized: np.ndarray, k: int, draws: int, rng: np.random.Generator
) -> float:
    """Mean realized improvement of the lowest-forecast seed over `draws` random k-subsets."""
    subsets = [rng.choice(len(forecast), size=k, replace=False) for _ in range(draws)]
    return float(np.mean([realized[idx[np.argmin(forecast[idx])]] for idx in subsets]))
