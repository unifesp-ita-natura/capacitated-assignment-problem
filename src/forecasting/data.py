"""Data-preparation helpers shared by the level and shape forecast strategies."""

from __future__ import annotations

from typing import Literal

import pandas as pd

CYCLE_TOTAL_COLUMNS = ["cycle_id", "sector", "cycle_total"]

DemandMetric = Literal["pedidos", "volumes", "itens"]

# metric -> (demanda_level total column, demanda_shape share column)
DEMAND_METRIC_COLUMNS: dict[DemandMetric, tuple[str, str]] = {
    "pedidos": ("total_pedidos", "share_pedidos"),
    "volumes": ("total_volumes", "share_volumes"),
    "itens": ("total_itens", "share_itens"),
}


def validate_demand_metric(metric: str) -> DemandMetric:
    """Raise if `metric` isn't one of the demand measures a demanda_level/demanda_shape
    table carries (only one makes sense as a forecast target at a time)."""
    if metric not in DEMAND_METRIC_COLUMNS:
        raise ValueError(
            f"Unknown demand metric {metric!r}, expected one of {sorted(DEMAND_METRIC_COLUMNS)}"
        )
    return metric


def build_cycle_totals(
    demand_level: pd.DataFrame, metric: DemandMetric = "pedidos"
) -> pd.DataFrame:
    """Each sector's total demand per cycle, from a demanda_level table
    (see src/processamento_dados/pipeline.py::build_demand_level), for one chosen metric."""
    total_column, _ = DEMAND_METRIC_COLUMNS[validate_demand_metric(metric)]
    return pd.DataFrame(
        {
            "cycle_id": demand_level["ciclo"].astype(int),
            "sector": demand_level["cd_setor"],
            "cycle_total": demand_level[total_column],
        }
    )


def build_shape_observations(
    demand_shape: pd.DataFrame, demand_level: pd.DataFrame, metric: DemandMetric = "pedidos"
) -> pd.DataFrame:
    """Each sector-cycle-day's demand share and integer day-in-cycle offset, from
    demanda_shape/demanda_level tables (see src/processamento_dados/pipeline.py),
    for one chosen metric."""
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
    """Rescale each sector's window-offset shares so they sum to 1."""
    totals = shape.groupby("sector")["order_share"].transform("sum")
    return shape.assign(order_share=shape["order_share"] / totals)


def to_calendar(
    day_offsets: pd.DataFrame, cycle_open_date: pd.Timestamp, start_day_by_sector: dict[str, int]
) -> pd.DataFrame:
    """Map (sector, offset) rows onto calendar dates.

    Each sector's window opens on its slot's business-day start (Mon-Fri), but
    orders within the window can land on any calendar day, weekends included.
    """
    frame = day_offsets.copy()
    starts = frame["sector"].map(start_day_by_sector)
    frame["day_in_cycle"] = starts + frame["offset"]
    window_start = starts.apply(lambda start_day: cycle_open_date + pd.offsets.BDay(start_day - 1))
    frame["order_date"] = window_start + pd.to_timedelta(frame["offset"], unit="D")
    return frame
