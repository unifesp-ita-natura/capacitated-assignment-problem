"""Generate synthetic orders, score every level/shape forecast strategy against a
held-out campanha, and report the best-performing combination."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from src.forecasting import data, level, scoring, sweep
from src.generate import generator

SEED = 7
SECTORS = [f"S{i:02d}" for i in range(1, 41)]
N_CAMPANHAS = 6
OUTPUT_DIR = "experiments/forecast_strategy_selection/outputs"

HALF_LIFE_GRID = (1.0, 2.0, 4.0)
SHRINKAGE_GRID = (0.1, 0.3, 0.5)


def level_strategies() -> dict[str, level.LevelStrategy]:
    """Every level strategy, minus the statsmodels-backed ones when that extra isn't installed."""
    if importlib.util.find_spec("statsmodels") is not None:
        return dict(level.LEVEL_STRATEGIES)
    statsmodels_only = {"holt_ets", "arima", "calendar_regression"}
    return {name: fn for name, fn in level.LEVEL_STRATEGIES.items() if name not in statsmodels_only}


def build_synthetic_orders(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[pd.Timestamp], dict]:
    """Synthetic historical orders, one campanha_start per campanha, and the as-is assignment."""
    assignment = generator.build_current_assignment(rng, SECTORS, generator.CURRENT_BLOCK_WEIGHTS)
    campanha_starts = [
        pd.Timestamp("2026-01-05") + pd.Timedelta(weeks=6 * i) for i in range(N_CAMPANHAS)
    ]
    orders = generator.generate_orders(rng, SECTORS, campanha_starts, assignment)
    return orders, campanha_starts, assignment


def split_history_and_holdout(
    orders: pd.DataFrame, holdout_campanha_id: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every campanha before the holdout as history, the holdout campanha's orders separately."""
    history = orders.loc[orders["campanha_id"].lt(holdout_campanha_id)]
    holdout = orders.loc[orders["campanha_id"].eq(holdout_campanha_id)]
    return history, holdout


def run() -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    rng = np.random.default_rng(SEED)
    orders, campanha_starts, assignment = build_synthetic_orders(rng)
    holdout_campanha_id = N_CAMPANHAS
    holdout_open_date = campanha_starts[-1]

    history, holdout = split_history_and_holdout(orders, holdout_campanha_id)
    campanhas = pd.DataFrame(
        {
            "campanha_id": range(1, N_CAMPANHAS + 1),
            "open_date": campanha_starts,
        }
    )

    campanha_totals = data.build_campanha_totals(history)
    shape_observations = data.build_shape_observations(history, campanha_totals)
    actual_campanha_total = data.build_campanha_totals(holdout).rename(
        columns={"campanha_total": "actual_campanha_total"}
    )[["sector", "actual_campanha_total"]]

    level_scoreboard = scoring.score_level_strategies(
        level_strategies(), campanha_totals, campanhas, actual_campanha_total
    )
    best_level_name = scoring.select_best_level_strategy(level_scoreboard)
    forecast_campanha_totals = level.forecast_campanha_total_with(
        level.LEVEL_STRATEGIES[best_level_name], campanha_totals, campanhas
    )

    shape_registry = sweep.build_shape_strategy_registry(HALF_LIFE_GRID, SHRINKAGE_GRID)
    shapes = {name: strategy(shape_observations) for name, strategy in shape_registry.items()}
    start_day_by_sector = {
        sector: generator.slot_start_day(block, sublock)
        for sector, (block, sublock) in assignment.items()
    }
    actual_daily = (
        holdout.groupby("order_date", as_index=False)["orders"]
        .sum()
        .rename(columns={"orders": "actual_orders"})
    )

    shape_scoreboard = scoring.score_shape_strategies(
        shapes, forecast_campanha_totals, start_day_by_sector, holdout_open_date, actual_daily
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
