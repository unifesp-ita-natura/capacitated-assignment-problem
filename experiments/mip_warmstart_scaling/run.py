"""Sweep sectors x cycles with the SA-warm-started MIP, skipping sizes after repeated timeouts."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.config.schema import PipelineConfig, RunConfig
from src.forecasting.dataset import load_demand_base
from src.pipeline.forecast_stage import forecast_horizon, plan_horizon
from src.scaling.store import append_row, load_completed
from src.scaling.sweep import Trial, sweep
from src.solver.heuristics import simulated_annealing
from src.solver.inputs import ProblemInstance
from src.solver.mip.block_assignment import solve_with_assignment
from src.solver.real_inputs import build_real_instance

CONFIG_PATH = Path("configs/pipeline.yaml")
OUTPUT_DIR = Path("experiments/mip_warmstart_scaling/outputs")
RESULTS_CSV = OUTPUT_DIR / "scaling.csv"

SECTOR_SWEEP = list(range(40, 1001, 40))
CYCLE_SWEEP = list(range(1, 7))
MAX_TIMEOUTS = 2
TIME_LIMIT_SECONDS = 3600.0


def _solve(instance: ProblemInstance, config: RunConfig, time_limit: float) -> Trial:
    """Solve `instance` with the MIP, warm-started from simulated annealing."""
    start = simulated_annealing.solve(instance, rng=random.Random(config.pipeline.seed)).assignment
    result, _ = solve_with_assignment(
        instance, solver=config.solver, time_limit_seconds=time_limit, warm_start=start
    )
    return Trial(result, len(instance.sectors))


def _point_runner(config: RunConfig, time_limit: float):
    """A `run_point(n_cycles, n_sectors)` that forecasts once per cycle count and caches it."""
    raw = load_demand_base(config.pipeline.base_csv)
    forecasts: dict[int, tuple] = {}

    def forecast_for(n_cycles: int):
        if n_cycles not in forecasts:
            pipeline = config.pipeline.model_copy(update={"horizon_cycles": n_cycles})
            plan = plan_horizon(raw, n_cycles, pipeline.history_cycles)
            forecasts[n_cycles] = (
                plan,
                forecast_horizon(raw, plan, pipeline, config.forecast.params),
            )
        return forecasts[n_cycles]

    def run_point(n_cycles: int, n_sectors: int) -> Trial:
        plan, forecast = forecast_for(n_cycles)
        pipeline: PipelineConfig = config.pipeline
        real = build_real_instance(
            forecast.daily,
            raw,
            plan.targets,
            as_is_cycle=forecast.training_cycles[-1],
            max_churn=pipeline.max_churn,
            capacity_multiplier=pipeline.capacity_multiplier,
            max_sectors=n_sectors,
        )
        trial = _solve(real.instance, config, time_limit)
        print(f"cycles={n_cycles} sectors={n_sectors}: {trial.result.termination_condition}")
        return trial

    return run_point


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_SECONDS)
    parser.add_argument("--max-timeouts", type=int, default=MAX_TIMEOUTS)
    parser.add_argument(
        "--overwrite", action="store_true", help="ignore and overwrite earlier results"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = _parse_args(argv)
    config = load_config(args.config)
    if args.overwrite:
        RESULTS_CSV.unlink(missing_ok=True)
    completed = load_completed(RESULTS_CSV)
    print(f"Resuming with {len(completed)} solved points from {RESULTS_CSV}")
    run_point = _point_runner(config, args.time_limit)
    rows = sweep(
        CYCLE_SWEEP,
        SECTOR_SWEEP,
        run_point,
        args.max_timeouts,
        completed,
        on_row=lambda row: append_row(RESULTS_CSV, row),
    )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
