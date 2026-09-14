"""Orchestrate the generate -> forecast -> solve pipeline to compare the MIP and SA solvers."""

from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from src.config import load_config
from src.forecasting import data, level, scoring
from src.forecasting import shape as shape_strategies
from src.forecasting.model import forecast_with_shape
from src.generate import generator
from src.persistence import SolveResult, write_results_csv
from src.solver.heuristics import simulated_annealing
from src.solver.mip.block_assignment import build_block_assignment_model, solve_block_assignment

METRIC = "pedidos"
RESULTS_FILENAME = "main_pipeline_comparison.csv"


def _build_sectors(n_sectors: int) -> list[str]:
    """Sector labels S01, S02, ... matching the generator's naming convention."""
    return [f"S{i:02d}" for i in range(1, n_sectors + 1)]


def _cycle_span_business_days(cycle_length: int) -> int:
    """Business days from a cycle's first slot opening to its last slot's window close.

    Mirrors `generator.CYCLE_SPAN` (which hardcodes `WINDOW_LENGTH`) so historical
    cycles built with a custom `cycle_length` still never overlap.
    """
    return len(generator.SLOTS) + cycle_length - 1


def _build_cycle_starts(cycle_length: int, n_cycles: int) -> list[pd.Timestamp]:
    """One non-overlapping start date per historical cycle, spaced by the cycle span."""
    base_date = pd.Timestamp("2026-01-05")  # Monday, arbitrary fixed anchor
    span = _cycle_span_business_days(cycle_length)
    return [base_date + pd.offsets.BDay(span * i) for i in range(n_cycles)]


def _next_cycle_start(cycle_starts: list[pd.Timestamp], cycle_length: int) -> pd.Timestamp:
    """Start date of the cycle immediately following the last historical one."""
    span = _cycle_span_business_days(cycle_length)
    return cycle_starts[-1] + pd.offsets.BDay(span)


def _generate_synthetic_demand(
    rng: np.random.Generator, sectors: list[str], cycle_length: int, n_cycles: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp], dict[str, tuple[int, int]]]:
    """Synthetic demanda_level/demanda_shape tables, cycle start dates, and the as-is assignment."""
    assignment = generator.build_current_assignment(rng, sectors, generator.CURRENT_BLOCK_WEIGHTS)
    cycle_starts = _build_cycle_starts(cycle_length, n_cycles)
    window_lengths = dict.fromkeys(range(1, n_cycles + 1), cycle_length)
    orders = generator.generate_orders(
        rng, sectors, cycle_starts, assignment, window_lengths=window_lengths
    )
    demand_level = generator.build_demand_level(orders)
    demand_shape = generator.build_demand_shape(orders)
    return demand_level, demand_shape, cycle_starts, assignment


def _split_history_and_holdout(
    demand: pd.DataFrame, holdout_cycle_id: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every cycle before the holdout as training history, the holdout cycle separately."""
    ciclo = demand["ciclo"].astype(int)
    return demand.loc[ciclo.lt(holdout_cycle_id)], demand.loc[ciclo.eq(holdout_cycle_id)]


def _score_and_select_strategies(
    demand_level: pd.DataFrame,
    demand_shape: pd.DataFrame,
    cycle_starts: list[pd.Timestamp],
    assignment: dict[str, tuple[int, int]],
    n_cycles: int,
) -> tuple[str, str]:
    """Best level and shape strategy names, chosen against the held-out last historical cycle."""
    holdout_cycle_id = n_cycles
    level_train, level_holdout = _split_history_and_holdout(demand_level, holdout_cycle_id)
    shape_train, shape_holdout = _split_history_and_holdout(demand_shape, holdout_cycle_id)
    cycles = pd.DataFrame({"cycle_id": range(1, n_cycles + 1), "open_date": cycle_starts})

    cycle_totals = data.build_cycle_totals(level_train, metric=METRIC)
    shape_observations = data.build_shape_observations(shape_train, level_train, metric=METRIC)
    actual_cycle_total = data.build_cycle_totals(level_holdout, metric=METRIC).rename(
        columns={"cycle_total": "actual_cycle_total"}
    )[["sector", "actual_cycle_total"]]

    level_scoreboard = scoring.score_level_strategies(
        level.LEVEL_STRATEGIES, cycle_totals, cycles, actual_cycle_total
    )
    best_level_name = scoring.select_best_level_strategy(level_scoreboard)

    forecast_cycle_totals = level.forecast_cycle_total_with(
        level.LEVEL_STRATEGIES[best_level_name], cycle_totals, cycles
    )
    start_day_by_sector = {
        sector: generator.slot_start_day(block, sublock)
        for sector, (block, sublock) in assignment.items()
    }
    total_column, _ = data.DEMAND_METRIC_COLUMNS[METRIC]
    actual_daily = (
        shape_holdout.groupby("data_pedido", as_index=False)[total_column]
        .sum()
        .rename(columns={"data_pedido": "order_date", total_column: "actual_orders"})
    )
    shapes = {
        name: strategy(shape_observations)
        for name, strategy in shape_strategies.SHAPE_STRATEGIES.items()
    }
    shape_scoreboard = scoring.score_shape_strategies(
        shapes, forecast_cycle_totals, start_day_by_sector, cycle_starts[-1], actual_daily
    )
    best_shape_name = scoring.select_best_shape_strategy(shape_scoreboard)
    return best_level_name, best_shape_name


def _forecast_next_cycle(
    demand_level: pd.DataFrame,
    demand_shape: pd.DataFrame,
    cycle_starts: list[pd.Timestamp],
    assignment: dict[str, tuple[int, int]],
    n_cycles: int,
    cycle_length: int,
    best_level_name: str,
    best_shape_name: str,
) -> pd.DataFrame:
    """Combined level+shape forecast for the next cycle, mapped onto calendar dates."""
    next_cycle_id = n_cycles + 1
    next_cycle_start = _next_cycle_start(cycle_starts, cycle_length)
    all_cycles = pd.DataFrame(
        {
            "cycle_id": [*range(1, n_cycles + 1), next_cycle_id],
            "open_date": [*cycle_starts, next_cycle_start],
        }
    )

    cycle_totals = data.build_cycle_totals(demand_level, metric=METRIC)
    shape_observations = data.build_shape_observations(demand_shape, demand_level, metric=METRIC)

    forecast_cycle_totals = level.forecast_cycle_total_with(
        level.LEVEL_STRATEGIES[best_level_name], cycle_totals, all_cycles
    )
    shape_forecast = shape_strategies.SHAPE_STRATEGIES[best_shape_name](shape_observations)
    combined = forecast_with_shape(shape_forecast, forecast_cycle_totals)

    start_day_by_sector = {
        sector: generator.slot_start_day(block, sublock)
        for sector, (block, sublock) in assignment.items()
    }
    return data.to_calendar(combined, next_cycle_start, start_day_by_sector)


def _build_id_maps(sectors: list[str]) -> tuple[dict[str, int], dict[int, tuple[int, int]]]:
    """Sector-label -> int id map, and combination-id -> (block, sublock) map."""
    sector_ids = {sector: i + 1 for i, sector in enumerate(sectors)}
    combo_slots = {i + 1: slot for i, slot in enumerate(generator.SLOTS)}
    return sector_ids, combo_slots


def _items_per_order_by_sector(demand_level: pd.DataFrame) -> dict[str, float]:
    """Each sector's empirical items-per-order ratio, from historical demand_level totals.

    `CD_CAPACITIES` is expressed in itens/dia while `forecast_orders` is a pedido
    (order) count; this ratio converts the latter into the former so demand and
    capacity are compared in the same unit.
    """
    totals = demand_level.groupby("cd_setor")[["total_itens", "total_pedidos"]].sum()
    return (totals["total_itens"] / totals["total_pedidos"]).to_dict()


def _forecast_items(
    combined_forecast: pd.DataFrame, items_per_order: dict[str, float]
) -> pd.DataFrame:
    """`combined_forecast` with a `forecast_items` column added (orders converted to itens)."""
    ratio = combined_forecast["sector"].map(items_per_order)
    return combined_forecast.assign(forecast_items=combined_forecast["forecast_orders"] * ratio)


def _projected_demand(
    combined_forecast: pd.DataFrame,
    sector_ids: dict[str, int],
    combo_slots: dict[int, tuple[int, int]],
) -> dict[tuple[int, int, int], float]:
    """Every (sector, day-in-cycle, combination) forecast quantity, in itens.

    A sector's forecast order-share curve is fixed; only where it *lands* in the
    cycle shifts with the (block, sublock) combination it is assigned to (each
    combination opens on a different day-in-cycle). This recomputes that landing
    day for every combination so the solvers can compare all of them.
    """
    projected_demand: dict[tuple[int, int, int], float] = {}
    for row in combined_forecast.itertuples(index=False):
        sector_id = sector_ids[row.sector]
        for combo_id, (block, sublock) in combo_slots.items():
            day_id = generator.slot_start_day(block, sublock) + row.offset
            projected_demand[(sector_id, day_id, combo_id)] = float(row.forecast_items)
    return projected_demand


def _build_daily_capacity(
    cd_codes: list[int], days: list[int], capacity_multiplier: float = 1.0
) -> dict[tuple[int, int], float]:
    """Flat per-day itens capacity for each CD, from its real daily throughput limit.

    `capacity_multiplier` is an optional headroom/derate factor applied on top of
    the real `CD_CAPACITIES` limit (1.0 leaves it unchanged).
    """
    return {
        (cd_code, day): capacity_multiplier * generator.CD_CAPACITIES[cd_code]
        for cd_code in cd_codes
        for day in days
    }


def _build_cd_sectors(
    sector_to_cd: dict[str, int], sector_ids: dict[str, int]
) -> dict[int, list[int]]:
    """CD code -> list of sector ids assigned to it."""
    cd_sectors: dict[int, list[int]] = {}
    for sector, cd_code in sector_to_cd.items():
        cd_sectors.setdefault(cd_code, []).append(sector_ids[sector])
    return cd_sectors


def _build_current_assignment_mip(
    assignment: dict[str, tuple[int, int]],
    sector_ids: dict[str, int],
    combo_ids: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    """As-is assignment as the sparse {(sector_id, combo_id): 1} mapping the MIP model expects."""
    return {(sector_ids[sector], combo_ids[slot]): 1 for sector, slot in assignment.items()}


def _build_current_assignment_sa(
    assignment: dict[str, tuple[int, int]],
    sector_ids: dict[str, int],
    combo_ids: dict[tuple[int, int], int],
) -> dict[int, int]:
    """As-is assignment as the dense {sector_id: combo_id} mapping the SA solver expects."""
    return {sector_ids[sector]: combo_ids[slot] for sector, slot in assignment.items()}


def _solve_mip(
    sector_ids_list: list[int],
    combo_ids_list: list[int],
    days: list[int],
    cd_sectors: dict[int, list[int]],
    daily_capacity: dict[tuple[int, int], float],
    projected_demand: dict[tuple[int, int, int], float],
    current_assignment_mip: dict[tuple[int, int], int],
    max_churn_sectors: float,
) -> SolveResult:
    """Build and solve the MIP block-assignment model with the open-source HiGHS backend."""
    model = build_block_assignment_model(
        sectors=sector_ids_list,
        combinations=combo_ids_list,
        days=days,
        cd_sectors=cd_sectors,
        daily_capacity=daily_capacity,
        projected_demand=projected_demand,
        current_assignment=current_assignment_mip,
        max_churn=max_churn_sectors,
    )
    start = time.perf_counter()
    pyomo_results = solve_block_assignment(model, solver="appsi_highs")
    wall_time_seconds = time.perf_counter() - start
    return SolveResult(
        model_name="mip_highs",
        solver="appsi_highs",
        status=str(pyomo_results.solver.status),
        termination_condition=str(pyomo_results.solver.termination_condition),
        objective=pyo.value(model.objective, exception=False),
        wall_time_seconds=wall_time_seconds,
    )


def _solve_simulated_annealing(
    sector_ids_list: list[int],
    combo_ids_list: list[int],
    days: list[int],
    cd_sectors: dict[int, list[int]],
    daily_capacity: dict[tuple[int, int], float],
    projected_demand: dict[tuple[int, int, int], float],
    current_assignment_sa: dict[int, int],
    max_churn_sectors: float,
    seed: int,
) -> SolveResult:
    """Run the simulated-annealing heuristic on the same block-assignment problem."""
    start = time.perf_counter()
    result = simulated_annealing.solve(
        sectors=sector_ids_list,
        combinations=combo_ids_list,
        days=days,
        cd_sectors=cd_sectors,
        daily_capacity=daily_capacity,
        projected_demand=projected_demand,
        current_assignment=current_assignment_sa,
        max_churn=max_churn_sectors,
        rng=random.Random(seed),
    )
    wall_time_seconds = time.perf_counter() - start
    return SolveResult(
        model_name="simulated_annealing",
        solver="simulated_annealing",
        status=result.stop_reason,
        termination_condition=result.stop_reason,
        objective=result.objective,
        wall_time_seconds=wall_time_seconds,
    )


def _print_comparison(mip_result: SolveResult, sa_result: SolveResult) -> None:
    """Short side-by-side summary of both solve outcomes."""
    print("\nSolver comparison:")
    for result in (mip_result, sa_result):
        print(
            f"  {result.model_name:<20} objective={result.objective!s:>12} "
            f"wall_time_s={result.wall_time_seconds:.3f} status={result.status}"
        )


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
    sectors = _build_sectors(n_sectors)

    demand_level, demand_shape, cycle_starts, assignment = _generate_synthetic_demand(
        rng, sectors, cycle_length, n_cycles
    )

    best_level_name, best_shape_name = _score_and_select_strategies(
        demand_level, demand_shape, cycle_starts, assignment, n_cycles
    )
    print(f"Selected strategies: level={best_level_name!r}, shape={best_shape_name!r}")

    calendar_forecast = _forecast_next_cycle(
        demand_level,
        demand_shape,
        cycle_starts,
        assignment,
        n_cycles,
        cycle_length,
        best_level_name,
        best_shape_name,
    )

    sector_ids, combo_slots = _build_id_maps(sectors)
    combo_ids = {slot: combo_id for combo_id, slot in combo_slots.items()}
    sector_ids_list = list(sector_ids.values())
    combo_ids_list = list(combo_slots.keys())

    items_per_order = _items_per_order_by_sector(demand_level)
    calendar_forecast = _forecast_items(calendar_forecast, items_per_order)

    projected_demand = _projected_demand(calendar_forecast, sector_ids, combo_slots)
    days = sorted({day for _, day, _ in projected_demand})

    sector_to_cd = generator.build_sector_cd_assignment(rng, sectors)
    cd_sectors = _build_cd_sectors(sector_to_cd, sector_ids)
    daily_capacity = _build_daily_capacity(list(cd_sectors), days, capacity_multiplier)

    current_assignment_mip = _build_current_assignment_mip(assignment, sector_ids, combo_ids)
    current_assignment_sa = _build_current_assignment_sa(assignment, sector_ids, combo_ids)
    max_churn_sectors = max_churn * n_sectors  # max_churn is a fraction of sectors allowed to move

    mip_result = _solve_mip(
        sector_ids_list,
        combo_ids_list,
        days,
        cd_sectors,
        daily_capacity,
        projected_demand,
        current_assignment_mip,
        max_churn_sectors,
    )
    sa_result = _solve_simulated_annealing(
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
    _print_comparison(mip_result, sa_result)


if __name__ == "__main__":
    main()
