"""Run SA, then a long SA-warm-started Gurobi MIP, and report the SA's certified gap."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyomo.environ as pyo

from src.calibration.mip import finite_or_none, relative_gap
from src.config import load_config
from src.forecasting.dataset import load_demand_base
from src.pipeline.forecast_stage import forecast_horizon, plan_horizon
from src.solver.heuristics import simulated_annealing
from src.solver.inputs import ProblemInstance
from src.solver.mip.block_assignment import apply_warm_start, build_block_assignment_model
from src.solver.real_inputs import build_real_instance

CONFIG_PATH = Path("configs/pipeline.yaml")
OUTPUT_DIR = Path("experiments/sa_certified_gap/outputs")
TIME_LIMIT_SECONDS = 7200.0
MEMORY_LIMIT_GB = 16.0


def certified_gap(sa_objective: float, lower_bound: float | None) -> dict[str, float | None]:
    """How far the SA objective can be from the optimum, given a valid MIP lower bound.

    `vs_incumbent` is (SA - LB) / SA, comparable to Gurobi's own gap; `vs_optimum` is
    (SA - LB) / LB, an upper bound on how much worse than optimal the SA can be.
    """
    vs_incumbent = relative_gap(lower_bound, sa_objective)
    if lower_bound is None or lower_bound <= 0:
        return {"vs_incumbent": vs_incumbent, "vs_optimum": None}
    return {"vs_incumbent": vs_incumbent, "vs_optimum": (sa_objective - lower_bound) / lower_bound}


def _build_instance(config_path: Path, n_cycles: int, max_sectors: int | None) -> ProblemInstance:
    config = load_config(config_path)
    pipeline = config.pipeline.model_copy(update={"horizon_cycles": n_cycles})
    raw = load_demand_base(pipeline.base_csv)
    plan = plan_horizon(raw, n_cycles, pipeline.history_cycles)
    forecast = forecast_horizon(raw, plan, pipeline, config.forecast.params)
    return build_real_instance(
        forecast.daily,
        raw,
        plan.targets,
        as_is_cycle=forecast.training_cycles[-1],
        max_churn=pipeline.max_churn,
        capacity_multiplier=pipeline.capacity_multiplier,
        max_sectors=max_sectors,
    ).instance


def _solve_mip(
    instance: ProblemInstance, start: dict[int, int], time_limit: float, log_path: Path
) -> dict:
    """Warm-started Gurobi solve; returns the incumbent and best bound."""
    model = build_block_assignment_model(instance)
    apply_warm_start(model, instance, start)
    solver = pyo.SolverFactory("gurobi")
    options = {"TimeLimit": time_limit, "MemLimit": MEMORY_LIMIT_GB, "LogFile": str(log_path)}
    results = solver.solve(model, options=options, warmstart=True, tee=True)
    upper = finite_or_none(results.problem.upper_bound)
    lower = finite_or_none(results.problem.lower_bound)
    return {
        "termination": str(results.solver.termination_condition),
        "objective": upper,
        "lower_bound": lower,
        "gap": relative_gap(lower, upper),
        "wall_time_seconds": results.solver.wall_time,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--max-sectors", type=int, default=None, help="default: all sectors")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_SECONDS)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    instance = _build_instance(args.config, args.cycles, args.max_sectors)
    sa = simulated_annealing.solve(instance, rng=random.Random(args.seed))
    record = {
        "n_cycles": args.cycles,
        "n_sectors": len(instance.sectors),
        "time_limit": args.time_limit,
        "sa": {"objective": sa.objective, "feasible": sa.feasible, "time": sa.time},
    }
    print(f"SA: objective={sa.objective:.4f} feasible={sa.feasible}")
    # Written before the long solve so the SA result survives a crash or interruption.
    (OUTPUT_DIR / "result.json").write_text(json.dumps(record, indent=2))
    record["mip"] = _solve_mip(instance, sa.assignment, args.time_limit, OUTPUT_DIR / "gurobi.log")
    record["certified_gap"] = certified_gap(sa.objective, record["mip"]["lower_bound"])
    (OUTPUT_DIR / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    main()
