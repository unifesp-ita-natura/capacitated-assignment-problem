"""Typed solve results and CSV persistence for comparing models/solvers side by side."""

from __future__ import annotations

import csv
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


def current_git_commit() -> str | None:
    """The current HEAD commit SHA, or None outside a git repo / without git installed."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return result.stdout.strip()


class SolveResult(BaseModel):
    model_name: str
    solver: str
    status: str
    termination_condition: str
    objective: float | None
    wall_time_seconds: float
    git_commit: str | None = Field(default_factory=current_git_commit)


class ForecastResult(BaseModel):
    model_name: str
    demand: dict[str, float]
    uncertainty: dict[str, float] | None
    forecast_horizon: tuple[date, date]  # (period_start, period_end) the demand values cover
    input_data_range: tuple[date, date]  # (history_start, history_end) used to train the model
    git_commit: str | None = Field(default_factory=current_git_commit)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("forecast_horizon", "input_data_range")
    @classmethod
    def _start_before_end(cls, value: tuple[date, date]) -> tuple[date, date]:
        start, end = value
        if start > end:
            raise ValueError(f"range start {start} must be <= end {end}")
        return value

    @field_validator("demand")
    @classmethod
    def _demand_non_negative(cls, value: dict[str, float]) -> dict[str, float]:
        negative = {k: v for k, v in value.items() if v < 0}
        if negative:
            raise ValueError(f"demand must be non-negative, got {negative}")
        return value

    @model_validator(mode="after")
    def _uncertainty_matches_demand(self) -> ForecastResult:
        if self.uncertainty is not None:
            extra = set(self.uncertainty) - set(self.demand)
            if extra:
                raise ValueError(f"uncertainty has keys not in demand: {extra}")
        return self


def write_results_csv(results: list[SolveResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SolveResult.model_fields))
        writer.writeheader()
        for result in results:
            writer.writerow(result.model_dump())
