"""Score level and shape forecast strategies against held-out actuals and rank them."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

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
    """Score one level-forecast strategy's per-sector totals against the actual cycle(s).

    Joins on `cycle_id` too when present in both frames, so a multi-cycle horizon's
    WMAPE aggregates over every (sector, cycle_id) pair, not just sector."""
    join_columns = ["sector", "cycle_id"] if "cycle_id" in actual_cycle_total.columns else "sector"
    comparison = forecast.merge(actual_cycle_total, on=join_columns)
    return metrics.mae_wmape(comparison, "forecast_cycle_total", "actual_cycle_total")


def score_level_strategies(
    strategies: dict[str, LevelStrategy],
    historical_cycle_totals: pd.DataFrame,
    cycles: pd.DataFrame,
    actual_cycle_total: pd.DataFrame,
    steps: Iterable[int] = (1,),
) -> pd.DataFrame:
    """Forecast and score every level strategy against the held-out cycle(s), sorted by WMAPE.

    A wide `steps` range (large `n_cycles_horizon` relative to `n_cycles_history`)
    shrinks the training window handed to each strategy, which can fall below what
    a statistical fit (Holt/ARIMA/calendar regression) needs to converge; such
    strategies are skipped from the scoreboard rather than crashing the whole run.
    """
    scores = []
    for name, strategy in strategies.items():
        try:
            forecast = forecast_cycle_total_with(strategy, historical_cycle_totals, cycles, steps)
            mae, wmape = score_level_strategy(forecast, actual_cycle_total)
        except Exception as error:  # noqa: BLE001 - third-party fit failures, not our bugs
            warnings.warn(f"Skipping level strategy {name!r}: {error}", stacklevel=2)
            continue
        scores.append({"strategy": name, "cycle_total_MAE": mae, "cycle_total_WMAPE": wmape})
    return metrics.build_scoreboard(scores, "cycle_total_WMAPE")


def select_best_level_strategy(level_scoreboard: pd.DataFrame) -> str:
    """Name of the level strategy with the lowest WMAPE on the held-out cycle."""
    return metrics.select_best(level_scoreboard, "cycle_total_WMAPE")


def score_shape_strategy(
    shape: pd.DataFrame,
    forecast_cycle_totals: pd.DataFrame,
    start_day_by_sector: dict[str, int],
    cycle_open_dates: dict[int, pd.Timestamp],
    actual_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """Forecast daily orders for one shape strategy and score it against the actual
    cycle(s) (dates across multiple holdout cycles don't collide, so no special-casing)."""
    forecast = model.forecast_with_shape(shape, forecast_cycle_totals)
    forecast = to_calendar(forecast, cycle_open_dates, start_day_by_sector)
    strategy_daily = forecast.groupby("order_date", as_index=False)["forecast_orders"].sum()

    comparison = strategy_daily.merge(actual_daily, how="outer", on="order_date").fillna(0)
    mae, wmape = metrics.mae_wmape(comparison, "forecast_orders", "actual_orders")
    return strategy_daily, mae, wmape


def score_shape_strategies(
    strategies: dict[str, pd.DataFrame],
    forecast_cycle_totals: pd.DataFrame,
    start_day_by_sector: dict[str, int],
    cycle_open_dates: dict[int, pd.Timestamp],
    actual_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Score every shape strategy against the held-out cycle(s), sorted by daily WMAPE."""
    scores = []
    for name, shape in strategies.items():
        _, mae, wmape = score_shape_strategy(
            shape, forecast_cycle_totals, start_day_by_sector, cycle_open_dates, actual_daily
        )
        scores.append({"strategy": name, "daily_MAE": mae, "daily_WMAPE": wmape})
    return metrics.build_scoreboard(scores, "daily_WMAPE")


def select_best_shape_strategy(shape_scoreboard: pd.DataFrame) -> str:
    """Name of the shape strategy with the lowest daily WMAPE on the held-out cycle."""
    return metrics.select_best(shape_scoreboard, "daily_WMAPE")


def build_holdout_cycle_ids(n_cycles_history: int, n_cycles_horizon: int) -> range:
    """The last `n_cycles_horizon` historical cycle ids, held out for backtesting."""
    if n_cycles_history <= n_cycles_horizon:
        raise ValueError(
            f"n_cycles_history ({n_cycles_history}) must exceed n_cycles_horizon "
            f"({n_cycles_horizon}) so at least one training cycle precedes the holdout window"
        )
    return range(n_cycles_history - n_cycles_horizon + 1, n_cycles_history + 1)


def score_and_select_strategies(
    demand_level: pd.DataFrame,
    demand_shape: pd.DataFrame,
    cycle_starts: list[pd.Timestamp],
    assignment: dict[str, tuple[int, int]],
    n_cycles_history: int,
    n_cycles_horizon: int = 1,
    metric: DemandMetric = "pedidos",
) -> tuple[str, str]:
    """Best level and shape strategy names, chosen against a multi-step backtest over
    the last `n_cycles_horizon` historical cycles (averaged WMAPE across that horizon)."""
    holdout_ids = build_holdout_cycle_ids(n_cycles_history, n_cycles_horizon)
    level_train, level_holdout = split_history_and_holdout(demand_level, set(holdout_ids))
    shape_train, shape_holdout = split_history_and_holdout(demand_shape, set(holdout_ids))
    cycles = pd.DataFrame({"cycle_id": range(1, n_cycles_history + 1), "open_date": cycle_starts})
    steps = range(1, n_cycles_horizon + 1)

    cycle_totals = build_cycle_totals(level_train, metric=metric)
    shape_observations = build_shape_observations(shape_train, level_train, metric=metric)
    actual_cycle_total = build_cycle_totals(level_holdout, metric=metric).rename(
        columns={"cycle_total": "actual_cycle_total"}
    )[["sector", "cycle_id", "actual_cycle_total"]]

    level_scoreboard = score_level_strategies(
        LEVEL_STRATEGIES, cycle_totals, cycles, actual_cycle_total, steps
    )
    best_level_name = select_best_level_strategy(level_scoreboard)

    forecast_cycle_totals = forecast_cycle_total_with(
        LEVEL_STRATEGIES[best_level_name], cycle_totals, cycles, steps
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
    cycle_open_dates = {cycle_id: cycle_starts[cycle_id - 1] for cycle_id in holdout_ids}
    shapes = {name: strategy(shape_observations) for name, strategy in SHAPE_STRATEGIES.items()}
    shape_scoreboard = score_shape_strategies(
        shapes, forecast_cycle_totals, start_day_by_sector, cycle_open_dates, actual_daily
    )
    best_shape_name = select_best_shape_strategy(shape_scoreboard)
    return best_level_name, best_shape_name
