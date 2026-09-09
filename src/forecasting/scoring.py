"""Score level and shape forecast strategies against held-out actuals and rank them."""

from __future__ import annotations

import pandas as pd

from src.forecasting import combine, metrics
from src.forecasting.data import to_calendar
from src.forecasting.level import LevelStrategy, forecast_campanha_total_with


def score_level_strategy(
    forecast: pd.DataFrame, actual_campanha_total: pd.DataFrame
) -> tuple[float, float]:
    """Score one level-forecast strategy's per-sector totals against the actual campanha."""
    comparison = forecast.merge(actual_campanha_total, on="sector")
    return metrics.mae_wmape(comparison, "forecast_campanha_total", "actual_campanha_total")


def score_level_strategies(
    strategies: dict[str, LevelStrategy],
    historical_campanha_totals: pd.DataFrame,
    campanhas: pd.DataFrame,
    actual_campanha_total: pd.DataFrame,
) -> pd.DataFrame:
    """Forecast and score every level strategy against the held-out campanha, sorted by WMAPE."""
    scores = []
    for name, strategy in strategies.items():
        forecast = forecast_campanha_total_with(strategy, historical_campanha_totals, campanhas)
        mae, wmape = score_level_strategy(forecast, actual_campanha_total)
        scores.append({"strategy": name, "campanha_total_MAE": mae, "campanha_total_WMAPE": wmape})
    return metrics.build_scoreboard(scores, "campanha_total_WMAPE")


def select_best_level_strategy(level_scoreboard: pd.DataFrame) -> str:
    """Name of the level strategy with the lowest WMAPE on the held-out campanha."""
    return metrics.select_best(level_scoreboard, "campanha_total_WMAPE")


def score_shape_strategy(
    shape: pd.DataFrame,
    forecast_campanha_totals: pd.DataFrame,
    start_day_by_sector: dict[str, int],
    campanha_open_date: pd.Timestamp,
    actual_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """Forecast daily orders for one shape strategy and score it against the actual campanha."""
    forecast = combine.forecast_with_shape(shape, forecast_campanha_totals)
    forecast = to_calendar(forecast, campanha_open_date, start_day_by_sector)
    strategy_daily = forecast.groupby("order_date", as_index=False)["forecast_orders"].sum()

    comparison = strategy_daily.merge(actual_daily, how="outer", on="order_date").fillna(0)
    mae, wmape = metrics.mae_wmape(comparison, "forecast_orders", "actual_orders")
    return strategy_daily, mae, wmape


def score_shape_strategies(
    strategies: dict[str, pd.DataFrame],
    forecast_campanha_totals: pd.DataFrame,
    start_day_by_sector: dict[str, int],
    campanha_open_date: pd.Timestamp,
    actual_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Score every shape strategy against the held-out campanha, sorted by daily WMAPE."""
    scores = []
    for name, shape in strategies.items():
        _, mae, wmape = score_shape_strategy(
            shape, forecast_campanha_totals, start_day_by_sector, campanha_open_date, actual_daily
        )
        scores.append({"strategy": name, "daily_MAE": mae, "daily_WMAPE": wmape})
    return metrics.build_scoreboard(scores, "daily_WMAPE")


def select_best_shape_strategy(shape_scoreboard: pd.DataFrame) -> str:
    """Name of the shape strategy with the lowest daily WMAPE on the held-out campanha."""
    return metrics.select_best(shape_scoreboard, "daily_WMAPE")
