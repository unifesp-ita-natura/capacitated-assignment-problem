"""Tests for the campanha-total (level) forecast strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.level import (
    extrapolate_linear_trend,
    forecast_campanha_total_with,
    level_linear_trend,
    level_naive_last,
)


def test_extrapolate_linear_trend_extends_a_perfect_trend():
    values = np.array([10.0, 20.0, 30.0, 40.0])

    assert extrapolate_linear_trend(values) == pytest.approx(50.0)


def test_extrapolate_linear_trend_clips_negative_forecasts_to_zero():
    values = np.array([30.0, 20.0, 10.0, 0.0])

    assert extrapolate_linear_trend(values) == 0.0


def test_extrapolate_linear_trend_returns_last_value_for_a_single_point():
    values = np.array([42.0])

    assert extrapolate_linear_trend(values) == 42.0


@pytest.fixture
def historical_campanha_totals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "campanha_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "sector": ["S1", "S1", "S1", "S1", "S2", "S2", "S2", "S2"],
            "campanha_total": [10.0, 20.0, 30.0, 40.0, 30.0, 20.0, 10.0, 0.0],
        }
    )


def test_level_naive_last_ignores_row_order(historical_campanha_totals):
    sector_totals = historical_campanha_totals[historical_campanha_totals["sector"].eq("S1")]
    shuffled = sector_totals.sample(frac=1, random_state=0)

    assert level_naive_last(shuffled, pd.DataFrame()) == 40.0


def test_level_linear_trend_matches_extrapolate_linear_trend(historical_campanha_totals):
    sector_totals = historical_campanha_totals[historical_campanha_totals["sector"].eq("S1")]

    assert level_linear_trend(sector_totals, pd.DataFrame()) == pytest.approx(50.0)


def test_forecast_campanha_total_with_applies_strategy_per_sector(historical_campanha_totals):
    forecast = forecast_campanha_total_with(
        level_linear_trend, historical_campanha_totals, pd.DataFrame()
    )

    result = forecast.set_index("sector")["forecast_campanha_total"]
    assert result["S1"] == pytest.approx(50.0)
    assert result["S2"] == 0.0


def test_level_holt_ets_forecasts_a_single_positive_value():
    pytest.importorskip("statsmodels")
    from src.forecasting.level import level_holt_ets

    values = pd.DataFrame({"campanha_id": [1, 2, 3, 4], "campanha_total": [10.0, 20.0, 30.0, 40.0]})

    forecast = level_holt_ets(values, pd.DataFrame())

    assert forecast >= 0.0


def test_level_arima_forecasts_a_single_non_negative_value():
    pytest.importorskip("statsmodels")
    from src.forecasting.level import level_arima

    values = pd.DataFrame({"campanha_id": [1, 2, 3, 4], "campanha_total": [10.0, 20.0, 30.0, 40.0]})

    forecast = level_arima(values, pd.DataFrame())

    assert forecast >= 0.0


def test_level_calendar_regression_forecasts_a_single_non_negative_value():
    pytest.importorskip("statsmodels")
    from src.forecasting.level import level_calendar_regression

    values = pd.DataFrame({"campanha_id": [1, 2, 3, 4], "campanha_total": [10.0, 20.0, 30.0, 40.0]})
    campanhas = pd.DataFrame(
        {
            "campanha_id": [1, 2, 3, 4, 5],
            "open_date": pd.date_range("2026-01-01", periods=5, freq="30D"),
        }
    )

    forecast = level_calendar_regression(values, campanhas)

    assert forecast >= 0.0
