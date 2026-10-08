"""Generate synthetic orders, score every level/shape forecast strategy against a
held-out cycle, and report the best-performing combination."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from src.forecasting import data, level, scoring, sweep
from src.generate import generator

SEED = 7
SECTORS = [f"S{i:02d}" for i in range(1, 41)]
N_CYCLES = 6
OUTPUT_DIR = "experiments/forecast_strategy_selection/outputs"

# which demanda_level/demanda_shape metric to forecast - only one makes sense as a
# target at a time. One of "pedidos", "volumes", "itens".
METRIC = "pedidos"

HALF_LIFE_GRID = (1.0, 2.0, 4.0)
SHRINKAGE_GRID = (0.1, 0.3, 0.5)


def level_strategies() -> dict[str, level.LevelStrategy]:
    """Every level strategy, minus the statsmodels-backed ones when that extra isn't installed."""
    if importlib.util.find_spec("statsmodels") is not None:
        return dict(level.LEVEL_STRATEGIES)
    statsmodels_only = {"holt_ets", "arima", "calendar_regression"}
    return {name: fn for name, fn in level.LEVEL_STRATEGIES.items() if name not in statsmodels_only}


def build_synthetic_demand(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp], dict]:
    """Synthetic demanda_level/demanda_shape tables, one cycle_start per cycle,
    and the as-is assignment."""
    assignment = generator.build_current_assignment(rng, SECTORS, generator.CURRENT_BLOCK_WEIGHTS)
    cycle_starts = [pd.Timestamp("2026-01-05") + pd.Timedelta(weeks=6 * i) for i in range(N_CYCLES)]
    orders = generator.generate_orders(rng, SECTORS, cycle_starts, assignment)
    demand_level = generator.build_demand_level(orders)
    demand_shape = generator.build_demand_shape(orders)
    return demand_level, demand_shape, cycle_starts, assignment


def split_history_and_holdout(
    demand: pd.DataFrame, holdout_cycle_id: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every cycle before the holdout as history, the holdout cycle separately.

    Works on either a demanda_level or demanda_shape table - both key cycles by
    the string `ciclo` column.
    """
    ciclo = demand["ciclo"].astype(int)
    history = demand.loc[ciclo.lt(holdout_cycle_id)]
    holdout = demand.loc[ciclo.eq(holdout_cycle_id)]
    return history, holdout


def run() -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    rng = np.random.default_rng(SEED)
    demand_level, demand_shape, cycle_starts, assignment = build_synthetic_demand(rng)
    holdout_cycle_id = N_CYCLES
    holdout_open_date = cycle_starts[-1]

    level_history, level_holdout = split_history_and_holdout(demand_level, holdout_cycle_id)
    shape_history, shape_holdout = split_history_and_holdout(demand_shape, holdout_cycle_id)
    cycles = pd.DataFrame(
        {
            "cycle_id": range(1, N_CYCLES + 1),
            "open_date": cycle_starts,
        }
    )

    cycle_totals = data.build_cycle_totals(level_history, metric=METRIC)
    shape_observations = data.build_shape_observations(shape_history, level_history, metric=METRIC)
    actual_cycle_total = data.build_cycle_totals(level_holdout, metric=METRIC).rename(
        columns={"cycle_total": "actual_cycle_total"}
    )[["sector", "actual_cycle_total"]]

    level_scoreboard = scoring.score_level_strategies(
        level_strategies(), cycle_totals, cycles, actual_cycle_total
    )
    best_level_name = scoring.select_best_level_strategy(level_scoreboard)
    forecast_cycle_totals = level.forecast_cycle_total_with(
        level.LEVEL_STRATEGIES[best_level_name], cycle_totals, cycles
    )

    shape_registry = sweep.build_shape_strategy_registry(HALF_LIFE_GRID, SHRINKAGE_GRID)
    shapes = {name: strategy(shape_observations) for name, strategy in shape_registry.items()}
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

    shape_scoreboard = scoring.score_shape_strategies(
        shapes, forecast_cycle_totals, start_day_by_sector, holdout_open_date, actual_daily
    )
    best_shape_name = scoring.select_best_shape_strategy(shape_scoreboard)

    return level_scoreboard, shape_scoreboard, best_level_name, best_shape_name


if __name__ == "__main__":
    import os

    level_scoreboard, shape_scoreboard, best_level_name, best_shape_name = run()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    level_scoreboard.to_csv(f"{OUTPUT_DIR}/level_scoreboard.csv")
    shape_scoreboard.to_csv(f"{OUTPUT_DIR}/shape_scoreboard.csv")

    print("Level strategy scoreboard:")
    print(level_scoreboard)
    print("\nShape strategy scoreboard:")
    print(shape_scoreboard)
    print(f"\nBest combination: level={best_level_name!r}, shape={best_shape_name!r}")
