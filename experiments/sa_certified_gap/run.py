"""Run SA, then a long SA-warm-started Gurobi MIP, and report the SA's certified gap."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.calibration.mip import relative_gap
from src.pipeline.real_instance import build_instance_for_cycles
from src.solver.heuristics import simulated_annealing
from src.solver.mip.gurobi_run import solve_logged

CONFIG_PATH = Path("configs/pipeline.yaml")
OUTPUT_DIR = Path("experiments/sa_certified_gap/outputs")
TIME_LIMIT_SECONDS = 7200.0
MIP_GAP = 1e-4  # Gurobi's default: effectively "run until the time limit"


def certified_gap(sa_objective: float, lower_bound: float | None) -> dict[str, float | None]:
    """How far the SA objective can be from the optimum, given a valid MIP lower bound.

    `vs_incumbent` is (SA - LB) / SA, comparable to Gurobi's own gap; `vs_optimum` is
    (SA - LB) / LB, an upper bound on how much worse than optimal the SA can be.
    """
    vs_incumbent = relative_gap(lower_bound, sa_objective)
    if lower_bound is None or lower_bound <= 0:
        return {"vs_incumbent": vs_incumbent, "vs_optimum": None}
    return {"vs_incumbent": vs_incumbent, "vs_optimum": (sa_objective - lower_bound) / lower_bound}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--max-sectors", type=int, default=None, help="default: all sectors")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_SECONDS)
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=MIP_GAP,
        help="stop once (best bound vs incumbent) relative gap falls below this, e.g. 0.01 = 1%%",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    instance = build_instance_for_cycles(args.config, args.cycles, args.max_sectors)
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
    record["mip_gap_target"] = args.mip_gap
    record["mip"], incumbent = solve_logged(
        instance, sa.assignment, args.time_limit, args.mip_gap, OUTPUT_DIR / "gurobi.log"
    )
    record["certified_gap"] = certified_gap(sa.objective, record["mip"]["lower_bound"])
    (OUTPUT_DIR / "result.json").write_text(json.dumps(record, indent=2))
    (OUTPUT_DIR / "assignment.json").write_text(json.dumps(incumbent, indent=2))
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    main()
