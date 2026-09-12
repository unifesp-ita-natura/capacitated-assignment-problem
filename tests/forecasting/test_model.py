"""Tests for combining level and shape forecasts into daily order counts."""

from __future__ import annotations

import pandas as pd

from src.forecasting.model import forecast_with_shape


def test_forecast_with_shape_rounds_shares_times_totals_to_whole_orders():
    shape = pd.DataFrame({"sector": ["S1", "S1"], "offset": [0, 1], "order_share": [0.25, 0.75]})
    forecast_totals = pd.DataFrame({"sector": ["S1"], "forecast_cycle_total": [20.0]})

    forecast = forecast_with_shape(shape, forecast_totals)

    assert forecast["forecast_orders"].tolist() == [5, 15]
