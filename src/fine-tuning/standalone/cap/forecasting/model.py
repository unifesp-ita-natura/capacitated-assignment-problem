"""Combine a level (cycle-total) forecast with a shape (within-window) forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cap.forecasting.data import build_cycle_totals, build_shape_observations, to_calendar
from cap.forecasting.level import LEVEL_STRATEGIES, forecast_cycle_total_with
from cap.forecasting.shape import SHAPE_STRATEGIES
from cap.generate.generator import future_cycle_starts, slot_start_day


def forecast_with_shape(shape, forecast_cycle_totals):
    forecast = shape.merge(forecast_cycle_totals, on="sector")
    forecast["forecast_orders"] = np.rint(
        forecast["order_share"] * forecast["forecast_cycle_total"]
    ).astype(int)
    return forecast


def forecast_future_cycles(
    demand_level,
    demand_shape,
    cycle_starts,
    assignment,
    n_cycles_history,
    cycle_length,
    best_level_name,
    best_shape_name,
    n_cycles_horizon=1,
    metric="pedidos",
):
    future_cycle_ids = [n_cycles_history + i for i in range(1, n_cycles_horizon + 1)]
    future_starts = future_cycle_starts(cycle_starts, cycle_length, n_cycles_horizon)
    all_cycles = pd.DataFrame(
        {
            "cycle_id": [*range(1, n_cycles_history + 1), *future_cycle_ids],
            "open_date": [*cycle_starts, *future_starts],
        }
    )
    cycle_totals = build_cycle_totals(demand_level, metric=metric)
    shape_observations = build_shape_observations(demand_shape, demand_level, metric=metric)
    forecast_cycle_totals = forecast_cycle_total_with(
        LEVEL_STRATEGIES[best_level_name],
        cycle_totals,
        all_cycles,
        steps=range(1, n_cycles_horizon + 1),
    )
    shape_forecast = SHAPE_STRATEGIES[best_shape_name](shape_observations)
    combined = forecast_with_shape(shape_forecast, forecast_cycle_totals)
    start_day_by_sector = {
        sector: slot_start_day(block, sublock) for sector, (block, sublock) in assignment.items()
    }
    cycle_open_dates = dict(zip(future_cycle_ids, future_starts))
    return to_calendar(combined, cycle_open_dates, start_day_by_sector)
