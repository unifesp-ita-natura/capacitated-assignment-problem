"""Tests for the per-series Prophet candidate (skipped where prophet isn't installed)."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("prophet")

import src.forecasting.candidates  # noqa: F401 - registers candidates into REGISTRY
from src.config.schema import ProphetParams
from src.forecasting.candidates.prophet import _future_dates, _prophet_fn
from src.forecasting.model import REGISTRY, InsufficientHistoryError


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2026-01-05", periods=len(values), freq="28D")
    return pd.Series(values, index=index)


def test_future_dates_step_by_the_median_observed_gap():
    history = pd.DatetimeIndex(pd.date_range("2026-01-05", periods=4, freq="28D"))

    future = _future_dates(history, 2)

    assert list(future) == [
        history[-1] + pd.Timedelta(days=28),
        history[-1] + pd.Timedelta(days=56),
    ]


def test_forecast_has_one_finite_non_negative_value_per_horizon_step():
    forecast = _prophet_fn(ProphetParams())(_series([100, 120, 110, 130, 125, 140, 135]), 2)

    assert len(forecast) == 2
    assert (forecast >= 0).all()


def test_short_series_is_skipped_rather_than_fitted():
    with pytest.raises(InsufficientHistoryError, match="cycles of history"):
        _prophet_fn(ProphetParams())(_series([100, 120, 110]), 1)


def test_registry_builds_the_prophet_candidate():
    assert REGISTRY.build(ProphetParams()).name == "prophet"
