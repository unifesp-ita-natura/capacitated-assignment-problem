"""Orchestrate the generate -> forecast -> solve pipeline to compare the MIP and SA solvers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import load_config
from src.forecasting.data import forecast_items, items_per_order_by_sector
from src.forecasting.model import forecast_future_cycles
from src.forecasting.scoring import score_and_select_strategies
from src.generate import generator
from src.persistence import write_results_csv
from src.solver import inputs as solver_inputs
from src.solver.heuristics.simulated_annealing import run_simulated_annealing
from src.solver.mip.block_assignment import run_block_assignment_mip

RESULTS_FILENAME = "main_pipeline_comparison.csv"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs/default.yaml"


def main(
    n_sectors: int = 40,
    cycle_length: int = 21,
    n_cycles_history: int = 6,
    n_cycles_horizon: int = 1,
    max_churn: float = 0.2,
    capacity_multiplier: float = 1.0,
    seed: int = 42,
    config_file: str | None = None,
) -> None:
    """Run the full generate -> forecast -> solve pipeline and compare both solvers.

    `n_cycles_history` is how many historical cycles to synthesize and forecast
    from; `n_cycles_horizon` is how many future cycles to forecast and check the
    (single, shared) assignment against.
    """
    rng = np.random.default_rng(seed)
    sectors = generator.build_sectors(n_sectors)

    demand_level, demand_shape, cycle_starts, assignment = generator.generate_synthetic_demand(
        rng, sectors, cycle_length, n_cycles_history
    )

    best_level_name, best_shape_name = score_and_select_strategies(
        demand_level, demand_shape, cycle_starts, assignment, n_cycles_history, n_cycles_horizon
    )
    print(f"Selected strategies: level={best_level_name!r}, shape={best_shape_name!r}")

    calendar_forecast = forecast_future_cycles(
        demand_level,
        demand_shape,
        cycle_starts,
        assignment,
        n_cycles_history,
        cycle_length,
        best_level_name,
        best_shape_name,
        n_cycles_horizon,
    )

    sector_ids, combo_slots = solver_inputs.build_id_maps(sectors)
    combo_ids = {slot: combo_id for combo_id, slot in combo_slots.items()}
    sector_ids_list = list(sector_ids.values())
    combo_ids_list = list(combo_slots.keys())

    items_per_order = items_per_order_by_sector(demand_level)
    calendar_forecast = forecast_items(calendar_forecast, items_per_order)

    cycle_span = generator.cycle_span_business_days(cycle_length)
    projected_demand = solver_inputs.build_projected_demand(
        calendar_forecast, sector_ids, combo_slots, cycle_span, n_cycles_history + 1
    )
    days = sorted({day for _, day, _ in projected_demand})
    days_by_cycle = solver_inputs.group_days_by_cycle(days, cycle_span)

    sector_to_cd = generator.build_sector_cd_assignment(rng, sectors)
    cd_sectors = solver_inputs.build_cd_sectors(sector_to_cd, sector_ids)
    daily_capacity = solver_inputs.build_daily_capacity(list(cd_sectors), days, capacity_multiplier)

    current_assignment_mip = solver_inputs.build_current_assignment_mip(
        assignment, sector_ids, combo_ids
    )
    current_assignment_sa = solver_inputs.build_current_assignment_sa(
        assignment, sector_ids, combo_ids
    )
    max_churn_sectors = max_churn * n_sectors  # max_churn is a fraction of sectors allowed to move

    mip_result = run_block_assignment_mip(
        sector_ids_list,
        combo_ids_list,
        days,
        cd_sectors,
        daily_capacity,
        projected_demand,
        current_assignment_mip,
        max_churn_sectors,
        days_by_cycle,
    )
    sa_result = run_simulated_annealing(
        sector_ids_list,
        combo_ids_list,
        days,
        cd_sectors,
        daily_capacity,
        projected_demand,
        current_assignment_sa,
        max_churn_sectors,
        days_by_cycle,
        seed,
    )

    config_file = config_file or DEFAULT_CONFIG
    config = load_config(DEFAULT_CONFIG)
    results_path = f"{config.paths.results}/{RESULTS_FILENAME}"
    write_results_csv([mip_result, sa_result], results_path)
    print(f"Wrote comparison results to {results_path}")
    print("\nSolver comparison:")
    for result in (mip_result, sa_result):
        print(
            f"  {result.model_name:<20} objective={result.objective!s:>12} "
            f"wall_time_s={result.wall_time_seconds:.3f} status={result.status}"
        )


if __name__ == "__main__":
    main()
