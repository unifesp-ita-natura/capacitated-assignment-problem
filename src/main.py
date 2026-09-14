"""Orchestrate the generate -> forecast -> solve pipeline to compare the MIP and SA solvers."""

from __future__ import annotations

import numpy as np

from src.config import load_config
from src.forecasting.data import forecast_items, items_per_order_by_sector
from src.forecasting.model import forecast_next_cycle
from src.forecasting.scoring import score_and_select_strategies
from src.generate import generator
from src.persistence import write_results_csv
from src.solver import inputs as solver_inputs
from src.solver.heuristics.simulated_annealing import run_simulated_annealing
from src.solver.mip.block_assignment import run_block_assignment_mip

RESULTS_FILENAME = "main_pipeline_comparison.csv"


def main(
    n_sectors: int = 40,
    cycle_length: int = 21,
    n_cycles: int = 6,
    max_churn: float = 0.2,
    capacity_multiplier: float = 1.0,
    seed: int = 42,
) -> None:
    """Run the full generate -> forecast -> solve pipeline and compare both solvers."""
    rng = np.random.default_rng(seed)
    sectors = generator.build_sectors(n_sectors)

    demand_level, demand_shape, cycle_starts, assignment = generator.generate_synthetic_demand(
        rng, sectors, cycle_length, n_cycles
    )

    best_level_name, best_shape_name = score_and_select_strategies(
        demand_level, demand_shape, cycle_starts, assignment, n_cycles
    )
    print(f"Selected strategies: level={best_level_name!r}, shape={best_shape_name!r}")

    calendar_forecast = forecast_next_cycle(
        demand_level,
        demand_shape,
        cycle_starts,
        assignment,
        n_cycles,
        cycle_length,
        best_level_name,
        best_shape_name,
    )

    sector_ids, combo_slots = solver_inputs.build_id_maps(sectors)
    combo_ids = {slot: combo_id for combo_id, slot in combo_slots.items()}
    sector_ids_list = list(sector_ids.values())
    combo_ids_list = list(combo_slots.keys())

    items_per_order = items_per_order_by_sector(demand_level)
    calendar_forecast = forecast_items(calendar_forecast, items_per_order)

    projected_demand = solver_inputs.build_projected_demand(
        calendar_forecast, sector_ids, combo_slots
    )
    days = sorted({day for _, day, _ in projected_demand})

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
        seed,
    )

    config = load_config("configs/default.yaml")
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
