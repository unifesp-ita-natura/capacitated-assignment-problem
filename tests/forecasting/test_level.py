"""Tests for the cycle-total (level) forecast strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.level import (
    extrapolate_linear_trend,
    forecast_cycle_total_with,
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
def historical_cycle_totals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cycle_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "sector": ["S1", "S1", "S1", "S1", "S2", "S2", "S2", "S2"],
            "cycle_total": [10.0, 20.0, 30.0, 40.0, 30.0, 20.0, 10.0, 0.0],
        }
    )


def test_level_naive_last_ignores_row_order(historical_cycle_totals):
    sector_totals = historical_cycle_totals[historical_cycle_totals["sector"].eq("S1")]
    shuffled = sector_totals.sample(frac=1, random_state=0)

    assert level_naive_last(shuffled, pd.DataFrame()) == 40.0


def test_level_linear_trend_matches_extrapolate_linear_trend(historical_cycle_totals):
    sector_totals = historical_cycle_totals[historical_cycle_totals["sector"].eq("S1")]

    assert level_linear_trend(sector_totals, pd.DataFrame()) == pytest.approx(50.0)


def test_forecast_cycle_total_with_applies_strategy_per_sector(historical_cycle_totals):
    forecast = forecast_cycle_total_with(
        level_linear_trend, historical_cycle_totals, pd.DataFrame()
    )

    result = forecast.set_index("sector")["forecast_cycle_total"]
    assert result["S1"] == pytest.approx(50.0)
    assert result["S2"] == 0.0
    assert forecast["cycle_id"].unique().tolist() == [5]


def test_forecast_cycle_total_with_produces_one_row_per_sector_per_step(historical_cycle_totals):
    forecast = forecast_cycle_total_with(
        level_linear_trend, historical_cycle_totals, pd.DataFrame(), steps=range(1, 4)
    )

    assert len(forecast) == 2 * 3  # 2 sectors x 3 steps
    assert sorted(forecast["cycle_id"].unique()) == [5, 6, 7]


def test_forecast_cycle_total_with_extrapolates_further_for_a_later_step(historical_cycle_totals):
    sector_totals = historical_cycle_totals[historical_cycle_totals["sector"].eq("S1")]

    forecast = forecast_cycle_total_with(
        level_linear_trend, sector_totals, pd.DataFrame(), steps=range(1, 3)
    )

    by_cycle = forecast.set_index("cycle_id")["forecast_cycle_total"]
    assert by_cycle[5] == pytest.approx(50.0)
    assert by_cycle[6] == pytest.approx(60.0)


def test_level_holt_ets_forecasts_a_single_positive_value():
    pytest.importorskip("statsmodels")
    from src.forecasting.level import level_holt_ets

    values = pd.DataFrame({"cycle_id": [1, 2, 3, 4], "cycle_total": [10.0, 20.0, 30.0, 40.0]})

    forecast = level_holt_ets(values, pd.DataFrame())

    assert forecast >= 0.0


def test_level_arima_forecasts_a_single_non_negative_value():
    pytest.importorskip("statsmodels")
    from src.forecasting.level import level_arima

    values = pd.DataFrame({"cycle_id": [1, 2, 3, 4], "cycle_total": [10.0, 20.0, 30.0, 40.0]})

    forecast = level_arima(values, pd.DataFrame())

    assert forecast >= 0.0


def test_level_calendar_regression_forecasts_a_single_non_negative_value():
    pytest.importorskip("statsmodels")
    from src.forecasting.level import level_calendar_regression

    values = pd.DataFrame({"cycle_id": [1, 2, 3, 4], "cycle_total": [10.0, 20.0, 30.0, 40.0]})
    cycles = pd.DataFrame(
        {
            "cycle_id": [1, 2, 3, 4, 5],
            "open_date": pd.date_range("2026-01-01", periods=5, freq="30D"),
        }
    )

    forecast = level_calendar_regression(values, cycles)

    assert forecast >= 0.0
