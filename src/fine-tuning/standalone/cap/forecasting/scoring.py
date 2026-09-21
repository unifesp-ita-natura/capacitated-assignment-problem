"""Score level and shape forecast strategies against held-out actuals and rank them."""

from __future__ import annotations

import warnings

import pandas as pd

from cap.forecasting import metrics, model
from cap.forecasting.data import (
    DEMAND_METRIC_COLUMNS,
    build_cycle_totals,
    build_shape_observations,
    split_history_and_holdout,
    to_calendar,
)
from cap.forecasting.level import LEVEL_STRATEGIES, forecast_cycle_total_with
from cap.forecasting.shape import SHAPE_STRATEGIES
from cap.generate.generator import slot_start_day


def score_level_strategy(forecast, actual_cycle_total):
    join_columns = ["sector", "cycle_id"] if "cycle_id" in actual_cycle_total.columns else "sector"
    comparison = forecast.merge(actual_cycle_total, on=join_columns)
    return metrics.mae_wmape(comparison, "forecast_cycle_total", "actual_cycle_total")


def score_level_strategies(
    strategies, historical_cycle_totals, cycles, actual_cycle_total, steps=(1,)
):
    scores = []
    for name, strategy in strategies.items():
        try:
            forecast = forecast_cycle_total_with(strategy, historical_cycle_totals, cycles, steps)
            mae, wmape = score_level_strategy(forecast, actual_cycle_total)
        except Exception as error:  # noqa: BLE001
            warnings.warn(f"Skipping level strategy {name!r}: {error}", stacklevel=2)
            continue
        scores.append({"strategy": name, "cycle_total_MAE": mae, "cycle_total_WMAPE": wmape})
    return metrics.build_scoreboard(scores, "cycle_total_WMAPE")


def select_best_level_strategy(level_scoreboard):
    return metrics.select_best(level_scoreboard, "cycle_total_WMAPE")


def score_shape_strategy(
    shape, forecast_cycle_totals, start_day_by_sector, cycle_open_dates, actual_daily
):
    forecast = model.forecast_with_shape(shape, forecast_cycle_totals)
    forecast = to_calendar(forecast, cycle_open_dates, start_day_by_sector)
    strategy_daily = forecast.groupby("order_date", as_index=False)["forecast_orders"].sum()
    comparison = strategy_daily.merge(actual_daily, how="outer", on="order_date").fillna(0)
    mae, wmape = metrics.mae_wmape(comparison, "forecast_orders", "actual_orders")
    return strategy_daily, mae, wmape


def score_shape_strategies(
    strategies, forecast_cycle_totals, start_day_by_sector, cycle_open_dates, actual_daily
):
    scores = []
    for name, shape in strategies.items():
        _, mae, wmape = score_shape_strategy(
            shape, forecast_cycle_totals, start_day_by_sector, cycle_open_dates, actual_daily
        )
        scores.append({"strategy": name, "daily_MAE": mae, "daily_WMAPE": wmape})
    return metrics.build_scoreboard(scores, "daily_WMAPE")


def select_best_shape_strategy(shape_scoreboard):
    return metrics.select_best(shape_scoreboard, "daily_WMAPE")


def build_holdout_cycle_ids(n_cycles_history: int, n_cycles_horizon: int) -> range:
    if n_cycles_history <= n_cycles_horizon:
        raise ValueError("n_cycles_history must exceed n_cycles_horizon")
    return range(n_cycles_history - n_cycles_horizon + 1, n_cycles_history + 1)


def score_and_select_strategies(
    demand_level,
    demand_shape,
    cycle_starts,
    assignment,
    n_cycles_history,
    n_cycles_horizon=1,
    metric="pedidos",
):
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
    return best_level_name, select_best_shape_strategy(shape_scoreboard)
