"""Tests for error metrics and scoreboard helpers."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.forecasting.metrics import (
    PRIMARY_METRIC,
    build_scoreboard,
    mae,
    mae_wmape,
    mase,
    rmse,
    select_best,
)


def test_mae_wmape_computes_mean_absolute_error_and_weighted_mape():
    comparison = pd.DataFrame({"forecast": [50.0, 0.0], "actual": [40.0, 10.0]})

    mae_val, wmape_val = mae_wmape(comparison, "forecast", "actual")

    assert mae_val == pytest.approx(10.0)
    assert wmape_val == pytest.approx(20.0 / 50.0)


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


def test_mae_is_mean_absolute_difference():
    assert mae([10, 20, 30], [12, 18, 33]) == (2 + 2 + 3) / 3


def test_mae_zero_when_predictions_match_exactly():
    assert mae([5, 5, 5], [5, 5, 5]) == 0.0


def test_rmse_penalizes_large_errors_more_than_mae():
    actual = [0, 0, 0, 0]
    predicted = [1, 1, 1, 10]  # one big miss, three small ones
    assert rmse(actual, predicted) > mae(actual, predicted)


def test_rmse_equals_mae_when_all_errors_are_equal():
    actual, predicted = [0, 0], [3, 3]
    assert rmse(actual, predicted) == mae(actual, predicted) == 3.0


def test_mase_below_one_means_better_than_naive_benchmark():
    result = mase([10, 10], [9, 11], in_sample_naive_mae=4.0)
    assert result == 0.25


def test_mase_is_infinite_when_naive_benchmark_had_zero_error():
    assert math.isinf(mase([10], [12], in_sample_naive_mae=0.0))


def test_primary_metric_is_mae():
    assert PRIMARY_METRIC == "mae"
