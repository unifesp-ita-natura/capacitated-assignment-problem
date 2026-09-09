"""Forecast each sector's fixed within-window order-share curve from historical orders."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

ShapeStrategy = Callable[[pd.DataFrame], pd.DataFrame]


def build_shape_observations(orders: pd.DataFrame, campanha_totals: pd.DataFrame) -> pd.DataFrame:
    """Historical orders joined with each sector-campanha's total, plus the resulting share."""
    observations = orders.merge(campanha_totals, on=["campanha_id", "sector"])
    observations["order_share"] = observations["orders"] / observations["campanha_total"]
    return observations


def normalize_shape(shape: pd.DataFrame) -> pd.DataFrame:
    """Rescale each sector's window-offset shares so they sum to 1."""
    totals = shape.groupby("sector")["order_share"].transform("sum")
    return shape.assign(order_share=shape["order_share"] / totals)


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


def build_shape_strategies(
    shape_observations: pd.DataFrame,
    half_life_campanhas: float = 2.0,
    shrinkage: float = 0.3,
) -> dict[str, pd.DataFrame]:
    """Every shape-forecast strategy's forecasted curve, keyed by strategy name."""
    plain_average = shape_plain_average(shape_observations)
    return {
        "plain_average": plain_average,
        "recency_weighted": shape_recency_weighted(shape_observations, half_life_campanhas),
        "median": shape_median(shape_observations),
        "last_campanha": shape_last_campanha(shape_observations),
        "shrinkage": shape_shrinkage(plain_average, shrinkage),
    }


def to_calendar(
    day_offsets: pd.DataFrame, campanha_open_date: pd.Timestamp, start_day_by_sector: dict[str, int]
) -> pd.DataFrame:
    """Map (sector, offset) rows onto calendar dates, given each sector's slot start day."""
    frame = day_offsets.copy()
    starts = frame["sector"].map(start_day_by_sector)
    frame["day_in_cycle"] = starts + frame["offset"]
    frame["order_date"] = campanha_open_date + pd.to_timedelta(frame["day_in_cycle"] - 1, unit="D")
    return frame


def forecast_with_shape(
    shape: pd.DataFrame, forecast_campanha_totals: pd.DataFrame
) -> pd.DataFrame:
    """Combine a within-window shape forecast with the campanha-total forecast into daily counts."""
    forecast = shape.merge(forecast_campanha_totals, on="sector")
    forecast["forecast_orders"] = np.rint(
        forecast["order_share"] * forecast["forecast_campanha_total"]
    ).astype(int)
    return forecast


def score_shape_strategy(
    shape: pd.DataFrame,
    forecast_campanha_totals: pd.DataFrame,
    start_day_by_sector: dict[str, int],
    campanha_open_date: pd.Timestamp,
    actual_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """Forecast daily orders for one shape strategy and score it against the actual campanha."""
    forecast = forecast_with_shape(shape, forecast_campanha_totals)
    forecast = to_calendar(forecast, campanha_open_date, start_day_by_sector)
    strategy_daily = forecast.groupby("order_date", as_index=False)["forecast_orders"].sum()

    comparison = strategy_daily.merge(actual_daily, how="outer", on="order_date").fillna(0)
    error = comparison["forecast_orders"] - comparison["actual_orders"]
    mae = float(error.abs().mean())
    wmape = float(error.abs().sum() / comparison["actual_orders"].sum())
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
    return pd.DataFrame(scores).set_index("strategy").sort_values("daily_WMAPE")


def select_best_shape_strategy(shape_scoreboard: pd.DataFrame) -> str:
    """Name of the shape strategy with the lowest daily WMAPE on the held-out campanha."""
    return str(shape_scoreboard["daily_WMAPE"].idxmin())
