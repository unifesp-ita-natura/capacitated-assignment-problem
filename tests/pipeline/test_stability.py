"""Tests for the realized-demand scoring and spread helpers of the seed-stability experiment."""

import pandas as pd

from src.pipeline.stability import SeedOutcome, realized_daily, spread


def test_realized_daily_swaps_forecast_for_actual_and_keeps_grid():
    date = pd.Timestamp("2026-01-02")
    keys = {"cd_setor": "S1", "estado": "SP", "cidade": "X", "CICLOS": "C1", "date": date}
    forecast = pd.DataFrame(
        [
            {**keys, "items_pred": 5.0},
            {**keys, "date": date + pd.Timedelta(days=1), "items_pred": 7.0},
        ]
    )
    raw = pd.DataFrame(
        [
            {
                "cd_setor": "S1",
                "estado": "SP",
                "cidade": "X",
                "CICLOS": "C1",
                "data_pedido": date,
                "total_itens_mascarado": 3,
            }
        ]
    )
    result = realized_daily(raw, forecast)
    assert result["items_pred"].tolist() == [3.0, 0.0]


def test_improvement_is_drop_relative_to_as_is():
    outcome = SeedOutcome(1, 100.0, 80.0, 50.0, 55.0, 3, 1.0)
    assert outcome.forecast_improvement_pct == 20.0
    assert outcome.realized_improvement_pct == -10.0


def test_spread_reports_cv():
    stats = spread(pd.Series([9.0, 11.0]))
    assert stats["mean"] == 10.0
    assert stats["min"] == 9.0 and stats["max"] == 11.0
    assert round(stats["cv_pct"], 2) == 14.14
