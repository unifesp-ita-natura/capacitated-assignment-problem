"""Data-preparation helpers shared by the level and shape forecast strategies."""

from __future__ import annotations

import pandas as pd

CAMPANHA_TOTAL_COLUMNS = ["campanha_id", "sector", "campanha_total"]


def build_campanha_totals(orders: pd.DataFrame) -> pd.DataFrame:
    """Each sector's total orders per campanha, from a raw orders table."""
    return (
        orders.groupby(["campanha_id", "sector"], as_index=False)["orders"]
        .sum()
        .rename(columns={"orders": "campanha_total"})
    )


def build_shape_observations(orders: pd.DataFrame, campanha_totals: pd.DataFrame) -> pd.DataFrame:
    """Historical orders joined with each sector-campanha's total, plus the resulting share."""
    observations = orders.merge(campanha_totals, on=["campanha_id", "sector"])
    observations["order_share"] = observations["orders"] / observations["campanha_total"]
    return observations


def normalize_shape(shape: pd.DataFrame) -> pd.DataFrame:
    """Rescale each sector's window-offset shares so they sum to 1."""
    totals = shape.groupby("sector")["order_share"].transform("sum")
    return shape.assign(order_share=shape["order_share"] / totals)


def to_calendar(
    day_offsets: pd.DataFrame, campanha_open_date: pd.Timestamp, start_day_by_sector: dict[str, int]
) -> pd.DataFrame:
    """Map (sector, offset) rows onto calendar dates.

    Each sector's window opens on its slot's business-day start (Mon-Fri), but
    orders within the window can land on any calendar day, weekends included.
    """
    frame = day_offsets.copy()
    starts = frame["sector"].map(start_day_by_sector)
    frame["day_in_cycle"] = starts + frame["offset"]
    window_start = starts.apply(
        lambda start_day: campanha_open_date + pd.offsets.BDay(start_day - 1)
    )
    frame["order_date"] = window_start + pd.to_timedelta(frame["offset"], unit="D")
    return frame
