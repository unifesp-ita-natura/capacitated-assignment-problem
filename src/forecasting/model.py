"""Combine a level (campanha-total) forecast with a shape (within-window) forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_with_shape(
    shape: pd.DataFrame, forecast_campanha_totals: pd.DataFrame
) -> pd.DataFrame:
    """Combine a within-window shape forecast with the campanha-total forecast into daily counts."""
    forecast = shape.merge(forecast_campanha_totals, on="sector")
    forecast["forecast_orders"] = np.rint(
        forecast["order_share"] * forecast["forecast_campanha_total"]
    ).astype(int)
    return forecast
