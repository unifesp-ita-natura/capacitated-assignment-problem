"""Score level and shape forecast strategies against held-out actuals and rank them."""

from __future__ import annotations

import pandas as pd

from src.forecasting import metrics, model
from src.forecasting.data import (
    DEMAND_METRIC_COLUMNS,
    DemandMetric,
    build_cycle_totals,
    build_shape_observations,
    split_history_and_holdout,
    to_calendar,
)
from src.forecasting.level import LEVEL_STRATEGIES, LevelStrategy, forecast_cycle_total_with
from src.forecasting.shape import SHAPE_STRATEGIES
from src.generate.generator import slot_start_day


def score_level_strategy(
    forecast: pd.DataFrame, actual_cycle_total: pd.DataFrame
) -> tuple[float, float]:
    """Score one level-forecast strategy's per-sector totals against the actual cycle."""
    comparison = forecast.merge(actual_cycle_total, on="sector")
    return metrics.mae_wmape(comparison, "forecast_cycle_total", "actual_cycle_total")


def score_level_strategies(
    strategies: dict[str, LevelStrategy],
    historical_cycle_totals: pd.DataFrame,
    cycles: pd.DataFrame,
    actual_cycle_total: pd.DataFrame,
) -> pd.DataFrame:
    """Forecast and score every level strategy against the held-out cycle, sorted by WMAPE."""
    scores = []
    for name, strategy in strategies.items():
        forecast = forecast_cycle_total_with(strategy, historical_cycle_totals, cycles)
        mae, wmape = score_level_strategy(forecast, actual_cycle_total)
        scores.append({"strategy": name, "cycle_total_MAE": mae, "cycle_total_WMAPE": wmape})
    return metrics.build_scoreboard(scores, "cycle_total_WMAPE")


def select_best_level_strategy(level_scoreboard: pd.DataFrame) -> str:
    """Name of the level strategy with the lowest WMAPE on the held-out cycle."""
    return metrics.select_best(level_scoreboard, "cycle_total_WMAPE")


def score_shape_strategy(
    shape: pd.DataFrame,
    forecast_cycle_totals: pd.DataFrame,
    start_day_by_sector: dict[str, int],
    cycle_open_date: pd.Timestamp,
    actual_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """Forecast daily orders for one shape strategy and score it against the actual cycle."""
    forecast = model.forecast_with_shape(shape, forecast_cycle_totals)
    forecast = to_calendar(forecast, cycle_open_date, start_day_by_sector)
    strategy_daily = forecast.groupby("order_date", as_index=False)["forecast_orders"].sum()

    comparison = strategy_daily.merge(actual_daily, how="outer", on="order_date").fillna(0)
    mae, wmape = metrics.mae_wmape(comparison, "forecast_orders", "actual_orders")
    return strategy_daily, mae, wmape


def score_shape_strategies(
    strategies: dict[str, pd.DataFrame],
    forecast_cycle_totals: pd.DataFrame,
    start_day_by_sector: dict[str, int],
    cycle_open_date: pd.Timestamp,
    actual_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Score every shape strategy against the held-out cycle, sorted by daily WMAPE."""
    scores = []
    for name, shape in strategies.items():
        _, mae, wmape = score_shape_strategy(
            shape, forecast_cycle_totals, start_day_by_sector, cycle_open_date, actual_daily
        )
        scores.append({"strategy": name, "daily_MAE": mae, "daily_WMAPE": wmape})
    return metrics.build_scoreboard(scores, "daily_WMAPE")


def select_best_shape_strategy(shape_scoreboard: pd.DataFrame) -> str:
    """Name of the shape strategy with the lowest daily WMAPE on the held-out cycle."""
    return metrics.select_best(shape_scoreboard, "daily_WMAPE")


def score_and_select_strategies(
    demand_level: pd.DataFrame,
    demand_shape: pd.DataFrame,
    cycle_starts: list[pd.Timestamp],
    assignment: dict[str, tuple[int, int]],
    n_cycles: int,
    metric: DemandMetric = "pedidos",
) -> tuple[str, str]:
    """Best level and shape strategy names, chosen against the held-out last historical cycle."""
    holdout_cycle_id = n_cycles
    level_train, level_holdout = split_history_and_holdout(demand_level, holdout_cycle_id)
    shape_train, shape_holdout = split_history_and_holdout(demand_shape, holdout_cycle_id)
    cycles = pd.DataFrame({"cycle_id": range(1, n_cycles + 1), "open_date": cycle_starts})

    cycle_totals = build_cycle_totals(level_train, metric=metric)
    shape_observations = build_shape_observations(shape_train, level_train, metric=metric)
    actual_cycle_total = build_cycle_totals(level_holdout, metric=metric).rename(
        columns={"cycle_total": "actual_cycle_total"}
    )[["sector", "actual_cycle_total"]]

    level_scoreboard = score_level_strategies(
        LEVEL_STRATEGIES, cycle_totals, cycles, actual_cycle_total
    )
    best_level_name = select_best_level_strategy(level_scoreboard)

    forecast_cycle_totals = forecast_cycle_total_with(
        LEVEL_STRATEGIES[best_level_name], cycle_totals, cycles
    )
    start_day_by_sector = {
        sector: slot_start_day(block, sublock) for sector, (block, sublock) in assignment.items()
    }
    total_column, _ = DEMAND_METRIC_COLUMNS[metric]
    actual_daily = (
        shape_holdout.groupby("data_pedido", as_index=False)[total_column]
        .sum()
        .rename(columns={"data_pedido": "order_date", total_column: "actual_orders"})
    )
    shapes = {name: strategy(shape_observations) for name, strategy in SHAPE_STRATEGIES.items()}
    shape_scoreboard = score_shape_strategies(
        shapes, forecast_cycle_totals, start_day_by_sector, cycle_starts[-1], actual_daily
    )
    best_shape_name = select_best_shape_strategy(shape_scoreboard)
    return best_level_name, best_shape_name
