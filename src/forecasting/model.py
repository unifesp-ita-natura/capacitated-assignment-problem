"""Combine a level (cycle-total) forecast with a shape (within-window) forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.data import (
    DemandMetric,
    build_cycle_totals,
    build_shape_observations,
    to_calendar,
)
from src.forecasting.level import LEVEL_STRATEGIES, forecast_cycle_total_with
from src.forecasting.shape import SHAPE_STRATEGIES
from src.generate.generator import next_cycle_start, slot_start_day


def forecast_with_shape(shape: pd.DataFrame, forecast_cycle_totals: pd.DataFrame) -> pd.DataFrame:
    """Combine a within-window shape forecast with the cycle-total forecast into daily counts."""
    forecast = shape.merge(forecast_cycle_totals, on="sector")
    forecast["forecast_orders"] = np.rint(
        forecast["order_share"] * forecast["forecast_cycle_total"]
    ).astype(int)
    return forecast


def forecast_next_cycle(
    demand_level: pd.DataFrame,
    demand_shape: pd.DataFrame,
    cycle_starts: list[pd.Timestamp],
    assignment: dict[str, tuple[int, int]],
    n_cycles: int,
    cycle_length: int,
    best_level_name: str,
    best_shape_name: str,
    metric: DemandMetric = "pedidos",
) -> pd.DataFrame:
    """Combined level+shape forecast for the next cycle, mapped onto calendar dates."""
    next_cycle_id = n_cycles + 1
    next_start = next_cycle_start(cycle_starts, cycle_length)
    all_cycles = pd.DataFrame(
        {
            "cycle_id": [*range(1, n_cycles + 1), next_cycle_id],
            "open_date": [*cycle_starts, next_start],
        }
    )

    cycle_totals = build_cycle_totals(demand_level, metric=metric)
    shape_observations = build_shape_observations(demand_shape, demand_level, metric=metric)

    forecast_cycle_totals = forecast_cycle_total_with(
        LEVEL_STRATEGIES[best_level_name], cycle_totals, all_cycles
    )
    shape_forecast = SHAPE_STRATEGIES[best_shape_name](shape_observations)
    combined = forecast_with_shape(shape_forecast, forecast_cycle_totals)

    start_day_by_sector = {
        sector: slot_start_day(block, sublock) for sector, (block, sublock) in assignment.items()
    }
    return to_calendar(combined, next_start, start_day_by_sector)
