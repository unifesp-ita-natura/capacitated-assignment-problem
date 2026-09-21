"""Minimal stand-in for src.persistence (pydantic not available offline)."""

from dataclasses import dataclass


@dataclass
class SolveResult:
    model_name: str
    solver: str
    status: str
    termination_condition: str
    objective: float | None
    wall_time_seconds: float
    git_commit: str | None = None
