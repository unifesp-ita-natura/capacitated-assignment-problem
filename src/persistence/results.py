"""Typed solve results and CSV persistence for comparing models/solvers side by side."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field


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


def write_results_csv(results: list[SolveResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SolveResult.model_fields))
        writer.writeheader()
        for result in results:
            writer.writerow(result.model_dump())
