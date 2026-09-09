"""Tests for the shared error-metric and scoreboard helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.metrics import build_scoreboard, mae_wmape, select_best


def test_mae_wmape_computes_mean_absolute_error_and_weighted_mape():
    comparison = pd.DataFrame({"forecast": [50.0, 0.0], "actual": [40.0, 10.0]})

    mae, wmape = mae_wmape(comparison, "forecast", "actual")

    assert mae == pytest.approx(10.0)
    assert wmape == pytest.approx(20.0 / 50.0)


def test_build_scoreboard_sorts_by_the_given_column_ascending():
    scores = [
        {"strategy": "b", "wmape": 0.4},
        {"strategy": "a", "wmape": 0.1},
    ]

    scoreboard = build_scoreboard(scores, "wmape")

    assert list(scoreboard.index) == ["a", "b"]
    assert scoreboard["wmape"].is_monotonic_increasing


def test_select_best_returns_the_lowest_value_strategy():
    scoreboard = pd.DataFrame(
        {"wmape": [0.4, 0.1, 0.2]},
        index=["naive_last", "linear_trend", "holt_ets"],
    )

    assert select_best(scoreboard, "wmape") == "linear_trend"
