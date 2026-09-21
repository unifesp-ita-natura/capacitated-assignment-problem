"""Data-preparation helpers shared by the level and shape forecast strategies."""

from __future__ import annotations

from typing import Literal

import pandas as pd

CYCLE_TOTAL_COLUMNS = ["cycle_id", "sector", "cycle_total"]

DemandMetric = Literal["pedidos", "volumes", "itens"]

DEMAND_METRIC_COLUMNS: dict[str, tuple[str, str]] = {
    "pedidos": ("total_pedidos", "share_pedidos"),
    "volumes": ("total_volumes", "share_volumes"),
    "itens": ("total_itens", "share_itens"),
}


def validate_demand_metric(metric: str) -> str:
    if metric not in DEMAND_METRIC_COLUMNS:
        raise ValueError(f"Unknown demand metric {metric!r}")
    return metric


def build_cycle_totals(demand_level: pd.DataFrame, metric: str = "pedidos") -> pd.DataFrame:
    total_column, _ = DEMAND_METRIC_COLUMNS[validate_demand_metric(metric)]
    return pd.DataFrame(
        {
            "cycle_id": demand_level["ciclo"].astype(int),
            "sector": demand_level["cd_setor"],
            "cycle_total": demand_level[total_column],
        }
    )


def split_history_and_holdout(demand: pd.DataFrame, holdout_cycle_ids: set[int]):
    ciclo = demand["ciclo"].astype(int)
    return demand.loc[ciclo.lt(min(holdout_cycle_ids))], demand.loc[ciclo.isin(holdout_cycle_ids)]


def build_shape_observations(
    demand_shape: pd.DataFrame, demand_level: pd.DataFrame, metric: str = "pedidos"
) -> pd.DataFrame:
    _, share_column = DEMAND_METRIC_COLUMNS[validate_demand_metric(metric)]
    cycle_durations = demand_level[["cd_setor", "ciclo", "cycle_duration"]]
    merged = demand_shape.merge(cycle_durations, on=["cd_setor", "ciclo"])
    return pd.DataFrame(
        {
            "cycle_id": merged["ciclo"].astype(int),
            "sector": merged["cd_setor"],
            "offset": (merged["relative_date"] * merged["cycle_duration"]).round().astype(int),
            "order_share": merged[share_column],
        }
    )


def normalize_shape(shape: pd.DataFrame) -> pd.DataFrame:
    totals = shape.groupby("sector")["order_share"].transform("sum")
    return shape.assign(order_share=shape["order_share"] / totals)


def to_calendar(day_offsets, cycle_open_dates, start_day_by_sector) -> pd.DataFrame:
    frame = day_offsets.copy()
    starts = frame["sector"].map(start_day_by_sector)
    frame["day_in_cycle"] = starts + frame["offset"]
    open_dates = frame["cycle_id"].map(cycle_open_dates)
    window_start = pd.Series(
        [
            open_date + pd.offsets.BDay(start_day - 1)
            for open_date, start_day in zip(open_dates, starts)
        ],
        index=frame.index,
    )
    frame["order_date"] = window_start + pd.to_timedelta(frame["offset"], unit="D")
    return frame


def items_per_order_by_sector(demand_level: pd.DataFrame) -> dict[str, float]:
    totals = demand_level.groupby("cd_setor")[["total_itens", "total_pedidos"]].sum()
    return (totals["total_itens"] / totals["total_pedidos"]).to_dict()


def forecast_items(combined_forecast: pd.DataFrame, items_per_order: dict[str, float]):
    ratio = combined_forecast["sector"].map(items_per_order)
    return combined_forecast.assign(forecast_items=combined_forecast["forecast_orders"] * ratio)
