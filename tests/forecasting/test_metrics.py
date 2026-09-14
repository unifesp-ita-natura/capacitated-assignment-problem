"""Tests for the error metrics forecasting/evaluation.py relies on."""

from __future__ import annotations

import math

from src.forecasting.metrics import PRIMARY_METRIC, mae, mase, rmse


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
    # candidate's own MAE is 1, naive in-sample MAE is 4 -> candidate is
    # much better than the benchmark it's scaled against.
    result = mase([10, 10], [9, 11], in_sample_naive_mae=4.0)
    assert result == 0.25


def test_mase_is_infinite_when_naive_benchmark_had_zero_error():
    assert math.isinf(mase([10], [12], in_sample_naive_mae=0.0))


def test_primary_metric_is_mae():
    # Hardcoded, single-source decision (see metrics.py docstring) — this
    # test exists so changing it is a deliberate, visible edit.
    assert PRIMARY_METRIC == "mae"
