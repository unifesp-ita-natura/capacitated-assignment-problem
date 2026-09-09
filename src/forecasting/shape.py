"""Forecast each sector's fixed within-window order-share curve from historical orders."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.forecasting.data import normalize_shape

ShapeStrategy = Callable[[pd.DataFrame], pd.DataFrame]


def shape_plain_average(shape_observations: pd.DataFrame) -> pd.DataFrame:
    """Within-window shape: unweighted average of order_share across historical campanhas."""
    shape = shape_observations.groupby(["sector", "offset"], as_index=False)["order_share"].mean()
    return normalize_shape(shape)


def shape_recency_weighted(
    shape_observations: pd.DataFrame, half_life_campanhas: float = 2.0
) -> pd.DataFrame:
    """Within-window shape: campanhas weighted by recency via exponential decay (EWMA)."""
    frame = shape_observations.copy()
    campanhas_ago = frame["campanha_id"].max() - frame["campanha_id"]
    frame["weight"] = 0.5 ** (campanhas_ago / half_life_campanhas)
    frame["weighted_share"] = frame["order_share"] * frame["weight"]
    grouped = frame.groupby(["sector", "offset"])
    shape = (
        (grouped["weighted_share"].sum() / grouped["weight"].sum())
        .rename("order_share")
        .reset_index()
    )
    return normalize_shape(shape)


def shape_median(shape_observations: pd.DataFrame) -> pd.DataFrame:
    """Within-window shape: median order_share across historical campanhas, robust to outliers."""
    shape = shape_observations.groupby(["sector", "offset"], as_index=False)["order_share"].median()
    return normalize_shape(shape)


def shape_last_campanha(shape_observations: pd.DataFrame) -> pd.DataFrame:
    """Within-window shape: only the most recent historical campanha's share, no averaging."""
    last_campanha_id = shape_observations["campanha_id"].max()
    columns = ["sector", "offset", "order_share"]
    shape = shape_observations.loc[shape_observations["campanha_id"].eq(last_campanha_id), columns]
    return normalize_shape(shape.reset_index(drop=True))


def shape_shrinkage(plain_average_shape: pd.DataFrame, shrinkage: float = 0.3) -> pd.DataFrame:
    """Within-window shape: blend each sector's own average curve with the cross-sector curve."""
    global_shape = (
        plain_average_shape.groupby("offset", as_index=False)["order_share"]
        .mean()
        .rename(columns={"order_share": "global_share"})
    )
    blended = plain_average_shape.merge(global_shape, on="offset")
    own_share = (1 - shrinkage) * blended["order_share"]
    global_contribution = shrinkage * blended["global_share"]
    blended["order_share"] = own_share + global_contribution
    return normalize_shape(blended[["sector", "offset", "order_share"]])


SHAPE_STRATEGIES: dict[str, ShapeStrategy] = {
    "plain_average": shape_plain_average,
    "median": shape_median,
    "last_campanha": shape_last_campanha,
}
