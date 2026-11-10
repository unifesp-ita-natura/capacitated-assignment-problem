"""Compare the HiGHS MIP with the HiGHS matheuristic at real size, same budget."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from src.pipeline.real_instance import build_instance_for_cycles
from src.solver.heuristics import simulated_annealing
from src.solver.matheuristics import RelaxAndFixParams, relax_and_fix
from src.solver.matheuristics.subproblem import pyomo_subproblem_solver
from src.solver.mip.block_assignment import solve_with_assignment

CONFIG_PATH = Path("configs/pipeline.yaml")
OUTPUT_DIR = Path("experiments/mip_anytime/outputs")
TIME_LIMIT_SECONDS = 600.0
SOLVER = "appsi_highs"
RUNS = ("mip_cold", "mip_sa_warm", "matheuristic_sa_warm")


def _mip(instance, start, time_limit: float) -> dict:
    result, _ = solve_with_assignment(
        instance, solver=SOLVER, time_limit_seconds=time_limit, warm_start=start
    )
    return {
        "objective": result.objective,
        "best_bound": result.best_bound,
        "gap": result.mip_gap,
        "termination": result.termination_condition,
        "wall_time_seconds": result.wall_time_seconds,
    }


def _matheuristic(instance, start, time_limit: float) -> dict:
    params = RelaxAndFixParams(total_time_limit=time_limit)
    found = relax_and_fix.solve(
        instance,
        params=params,
        solve_subproblem=pyomo_subproblem_solver(SOLVER, params.mip_rel_gap),
        warm_start=start,
    )
    return {
        "objective": found.objective if found.feasible else None,
        "termination": found.stop_reason,
        "wall_time_seconds": found.time,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--max-sectors", type=int, default=None, help="default: all sectors")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_SECONDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    instance = build_instance_for_cycles(args.config, args.cycles, args.max_sectors)
    started = time.perf_counter()
    sa = simulated_annealing.solve(instance, rng=random.Random(args.seed))
    sa_seconds = time.perf_counter() - started
    runners = {
        "mip_cold": lambda: _mip(instance, None, args.time_limit),
        "mip_sa_warm": lambda: _mip(instance, sa.assignment, args.time_limit),
        "matheuristic_sa_warm": lambda: _matheuristic(instance, sa.assignment, args.time_limit),
    }
    record = {"n_sectors": len(instance.sectors), "sa": sa.objective, "sa_seconds": sa_seconds}
    for name in RUNS:
        path = OUTPUT_DIR / f"highs_{name}.json"
        if not path.exists() or args.overwrite:
            path.write_text(json.dumps(runners[name](), indent=2))
        record[name] = json.loads(path.read_text())
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    main()
