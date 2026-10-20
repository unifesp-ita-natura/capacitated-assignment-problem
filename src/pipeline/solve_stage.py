"""Run the chosen solver family on an instance and return its result with the assignment."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import NamedTuple

from src.config.schema import RunConfig
from src.persistence import SolveResult
from src.solver.heuristics import simulated_annealing
from src.solver.inputs import ProblemInstance
from src.solver.matheuristics import RelaxAndFixParams, relax_and_fix
from src.solver.matheuristics.subproblem import pyomo_subproblem_solver
from src.solver.mip.block_assignment import solve_with_assignment

MATHEURISTIC_SOLVER = "appsi_gurobi"


class Solved(NamedTuple):
    """A solve's metadata and the sector -> combination map it found (None if infeasible)."""

    result: SolveResult
    assignment: dict[int, int] | None


def _mip(instance: ProblemInstance, config: RunConfig) -> Solved:
    return Solved(*solve_with_assignment(instance, solver=config.solver))


def _sa(instance: ProblemInstance, config: RunConfig) -> Solved:
    start = time.perf_counter()
    found = simulated_annealing.solve(instance, rng=random.Random(config.pipeline.seed))
    result = SolveResult(
        model_name="simulated_annealing",
        solver="simulated_annealing",
        status=found.stop_reason,
        termination_condition=found.stop_reason,
        objective=found.objective,
        wall_time_seconds=time.perf_counter() - start,
    )
    return Solved(result, found.assignment)


def _matheuristic(instance: ProblemInstance, config: RunConfig) -> Solved:
    params = RelaxAndFixParams()
    found = relax_and_fix.solve(
        instance,
        params=params,
        solve_subproblem=pyomo_subproblem_solver(MATHEURISTIC_SOLVER, params.mip_rel_gap),
    )
    result = SolveResult(
        model_name="relax_and_fix",
        solver=MATHEURISTIC_SOLVER,
        status=found.stop_reason,
        termination_condition="feasible" if found.feasible else "infeasible",
        objective=found.objective if found.feasible else None,
        wall_time_seconds=found.time,
    )
    return Solved(result, found.assignment)


SOLVERS: dict[str, Callable[[ProblemInstance, RunConfig], Solved]] = {
    "mip": _mip,
    "matheuristic": _matheuristic,
    "sa": _sa,
}


def run_solver(family: str, instance: ProblemInstance, config: RunConfig) -> Solved:
    """Dispatch to the `mip`, `matheuristic` or `sa` runner."""
    return SOLVERS[family](instance, config)
