"""Forecast each sector's fixed within-window order-share curve from historical orders."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from cap.forecasting.data import normalize_shape

ShapeStrategy = Callable[[pd.DataFrame], pd.DataFrame]


def shape_plain_average(shape_observations: pd.DataFrame) -> pd.DataFrame:
    shape = shape_observations.groupby(["sector", "offset"], as_index=False)["order_share"].mean()
    return normalize_shape(shape)


def shape_median(shape_observations: pd.DataFrame) -> pd.DataFrame:
    shape = shape_observations.groupby(["sector", "offset"], as_index=False)["order_share"].median()
    return normalize_shape(shape)


def shape_last_cycle(shape_observations: pd.DataFrame) -> pd.DataFrame:
    last_cycle_id = shape_observations["cycle_id"].max()
    columns = ["sector", "offset", "order_share"]
    shape = shape_observations.loc[shape_observations["cycle_id"].eq(last_cycle_id), columns]
    return normalize_shape(shape.reset_index(drop=True))


SHAPE_STRATEGIES: dict[str, ShapeStrategy] = {
    "plain_average": shape_plain_average,
    "median": shape_median,
    "last_cycle": shape_last_cycle,
}
