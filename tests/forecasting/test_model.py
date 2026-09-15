"""Tests for combining level and shape forecasts into daily order counts."""

from __future__ import annotations

import pandas as pd

from src.forecasting.model import forecast_future_cycles, forecast_with_shape


def test_forecast_with_shape_rounds_shares_times_totals_to_whole_orders():
    shape = pd.DataFrame({"sector": ["S1", "S1"], "offset": [0, 1], "order_share": [0.25, 0.75]})
    forecast_totals = pd.DataFrame({"sector": ["S1"], "forecast_cycle_total": [20.0]})

    forecast = forecast_with_shape(shape, forecast_totals)

    assert forecast["forecast_orders"].tolist() == [5, 15]


def _build_synthetic_demand() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two cycles of demand_level/demand_shape rows for a single sector."""
    demand_level = pd.DataFrame(
        {
            "cd_setor": ["S1", "S1"],
            "ciclo": ["1", "2"],
            "cycle_duration": [3, 3],
            "total_pedidos": [30, 42],
            "total_volumes": [60, 84],
            "total_itens": [90, 126],
        }
    )
    demand_shape = pd.DataFrame(
        {
            "cd_setor": ["S1"] * 6,
            "ciclo": ["1", "1", "1", "2", "2", "2"],
            "relative_date": [0 / 3, 1 / 3, 2 / 3, 0 / 3, 1 / 3, 2 / 3],
            "share_pedidos": [0.2, 0.3, 0.5, 0.2, 0.3, 0.5],
        }
    )
    return demand_level, demand_shape


def test_forecast_future_cycles_returns_calendar_mapped_forecast_for_expected_sectors():
    demand_level, demand_shape = _build_synthetic_demand()
    cycle_starts = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-02")]
    assignment = {"S1": (1, 1)}

    forecast = forecast_future_cycles(
        demand_level=demand_level,
        demand_shape=demand_shape,
        cycle_starts=cycle_starts,
        assignment=assignment,
        n_cycles_history=2,
        cycle_length=3,
        best_level_name="naive_last",
        best_shape_name="plain_average",
    )

    assert {"forecast_orders", "order_date"} <= set(forecast.columns)
    assert set(forecast["sector"].unique()) == {"S1"}
    assert forecast["cycle_id"].unique().tolist() == [3]


def test_forecast_future_cycles_covers_every_cycle_in_the_horizon():
    demand_level, demand_shape = _build_synthetic_demand()
    cycle_starts = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-02")]
    assignment = {"S1": (1, 1)}

    forecast = forecast_future_cycles(
        demand_level=demand_level,
        demand_shape=demand_shape,
        cycle_starts=cycle_starts,
        assignment=assignment,
        n_cycles_history=2,
        cycle_length=3,
        best_level_name="naive_last",
        best_shape_name="plain_average",
        n_cycles_horizon=3,
    )

    assert sorted(forecast["cycle_id"].unique()) == [3, 4, 5]
