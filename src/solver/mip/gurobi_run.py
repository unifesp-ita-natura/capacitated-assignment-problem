"""Solve the block-assignment MIP with Gurobi (cold or warm-started), logging to a file."""

from __future__ import annotations

from pathlib import Path

import pyomo.environ as pyo

from src.calibration.mip import finite_or_none, relative_gap
from src.solver.inputs import ProblemInstance
from src.solver.mip.block_assignment import (
    apply_warm_start,
    build_block_assignment_model,
    extract_assignment,
)

MEMORY_LIMIT_GB = 16.0


def solve_logged(
    instance: ProblemInstance,
    start: dict[int, int] | None,
    time_limit: float,
    mip_gap: float,
    log_path: Path,
) -> tuple[dict, dict[int, int]]:
    """Gurobi solve stopped at `time_limit` or relative gap `mip_gap`, warm-started if `start`.

    Returns the incumbent/bound summary and the incumbent assignment (sector -> combination).
    """
    model = build_block_assignment_model(instance)
    if start is not None:
        apply_warm_start(model, instance, start)
    options = {
        "TimeLimit": time_limit,
        "MIPGap": mip_gap,
        "MemLimit": MEMORY_LIMIT_GB,
        "LogFile": str(log_path),
    }
    results = pyo.SolverFactory("gurobi").solve(
        model, options=options, warmstart=start is not None, tee=True
    )
    upper = finite_or_none(results.problem.upper_bound)
    lower = finite_or_none(results.problem.lower_bound)
    summary = {
        "termination": str(results.solver.termination_condition),
        "objective": upper,
        "lower_bound": lower,
        "gap": relative_gap(lower, upper),
        "wall_time_seconds": results.solver.wall_time,
    }
    return summary, extract_assignment(model)
