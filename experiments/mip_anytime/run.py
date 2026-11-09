"""Record the Gurobi MIP's incumbent and bound over time at real size, cold and SA-warm-started."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.pipeline.real_instance import build_instance_for_cycles
from src.solver.heuristics import simulated_annealing
from src.solver.mip.gurobi_run import solve_logged
from src.solver.mip.gurobi_trace import parse_trace, state_at

CONFIG_PATH = Path("configs/pipeline.yaml")
OUTPUT_DIR = Path("experiments/mip_anytime/outputs")
TIME_LIMIT_SECONDS = 600.0
CHECKPOINTS = (10, 30, 60, 120, 300, 600)
MODES = ("cold", "sa_warm")


def checkpoint_table(log_text: str) -> dict[int, dict[str, float] | None]:
    """Incumbent, bound and gap to the bound at each checkpoint second (None if not logged yet)."""
    trace = parse_trace(log_text)
    table = {}
    for second in CHECKPOINTS:
        point = state_at(trace, second)
        table[second] = (
            None
            if point is None
            else {
                "incumbent": point.incumbent,
                "bound": point.bound,
                "gap_to_bound": (point.incumbent - point.bound) / point.bound,
            }
        )
    return table


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
    sa = simulated_annealing.solve(instance, rng=random.Random(args.seed))
    starts = {"cold": None, "sa_warm": sa.assignment}
    record = {"n_sectors": len(instance.sectors), "n_cycles": args.cycles, "sa": sa.objective}
    for mode in MODES:
        path = OUTPUT_DIR / f"{mode}.json"
        if path.exists() and not args.overwrite:
            record[mode] = json.loads(path.read_text())
            continue
        log_path = OUTPUT_DIR / f"{mode}.log"
        summary, _ = solve_logged(instance, starts[mode], args.time_limit, 1e-4, log_path)
        summary["checkpoints"] = checkpoint_table(log_path.read_text())
        path.write_text(json.dumps(summary, indent=2))
        record[mode] = summary
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    main()
