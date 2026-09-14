"""Tests for the ARIMA/SARIMA candidate and its history guards."""

from __future__ import annotations

import pandas as pd
import pytest

import src.forecasting.candidates  # noqa: F401 - registers candidates into REGISTRY
from src.config.schema import ArimaParams
from src.forecasting.candidates.arima import _arima_fn, _candidate_name, _minimum_observations
from src.forecasting.model import REGISTRY, InsufficientHistoryError


def test_minimum_observations_grows_with_the_order():
    simple = _minimum_observations(ArimaParams(order=(1, 0, 0)))
    richer = _minimum_observations(ArimaParams(order=(2, 1, 2)))

    assert richer > simple


def test_minimum_observations_accounts_for_the_seasonal_period():
    # A seasonal difference at period 14 consumes 14 observations before
    # anything can be estimated — which is why seasonal orders can't be used
    # on a base with a single year of cycles (see docs/agent-log).
    seasonal = _minimum_observations(ArimaParams(order=(1, 0, 0), seasonal_order=(0, 1, 1, 14)))

    assert seasonal > 14


def test_short_series_is_skipped_rather_than_fitted():
    predict = _arima_fn(ArimaParams(order=(2, 1, 2)))

    with pytest.raises(InsufficientHistoryError, match="cycles of history"):
        predict(pd.Series([100.0, 110.0, 120.0]), 1)


def test_forecast_is_never_negative():
    # A steeply declining series extrapolates below zero unless clipped, and
    # negative item demand is meaningless (ForecastResult rejects it too).
    predict = _arima_fn(ArimaParams(order=(1, 1, 0)))

    forecast = predict(pd.Series([1000.0, 800.0, 600.0, 400.0, 200.0, 100.0]), 3)

    assert (forecast >= 0).all()


def test_candidate_is_named_arima_without_a_seasonal_order():
    assert _candidate_name(ArimaParams(order=(1, 0, 0))) == "arima(1,0,0)"


def test_candidate_is_named_sarima_with_a_seasonal_order():
    name = _candidate_name(ArimaParams(order=(1, 0, 0), seasonal_order=(0, 1, 1, 14)))

    assert name == "sarima(1,0,0)(0,1,1,14)"


def test_a_fit_failure_is_translated_into_a_skipped_sector(monkeypatch):
    # statsmodels failing to fit one sector is the model declining it, not a
    # bug — the harness should count it as missing, not crash the run.
    def explode(*args, **kwargs):
        raise ValueError("LU decomposition error")

    monkeypatch.setattr("src.forecasting.candidates.arima._fit_and_forecast", explode)
    predict = _arima_fn(ArimaParams(order=(1, 0, 0)))

    with pytest.raises(InsufficientHistoryError, match="could not fit"):
        predict(pd.Series([100.0, 110.0, 120.0, 130.0, 140.0, 150.0]), 1)


def test_an_unexpected_error_still_propagates(monkeypatch):
    # Only the known statsmodels failure modes are translated; anything else
    # is a real bug and must not be swallowed as "skip this sector".
    def explode(*args, **kwargs):
        raise RuntimeError("something genuinely broken")

    monkeypatch.setattr("src.forecasting.candidates.arima._fit_and_forecast", explode)
    predict = _arima_fn(ArimaParams(order=(1, 0, 0)))

    with pytest.raises(RuntimeError, match="genuinely broken"):
        predict(pd.Series([100.0, 110.0, 120.0, 130.0, 140.0, 150.0]), 1)


def test_a_non_finite_forecast_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "src.forecasting.candidates.arima._fit_and_forecast",
        lambda *args, **kwargs: pd.Series([float("nan")]),
    )
    predict = _arima_fn(ArimaParams(order=(1, 0, 0)))

    with pytest.raises(InsufficientHistoryError, match="non-finite"):
        predict(pd.Series([100.0, 110.0, 120.0, 130.0, 140.0, 150.0]), 1)


def test_registry_builds_the_arima_candidate_from_config():
    candidate = REGISTRY.build(ArimaParams(order=(1, 0, 0)))

    assert candidate.name == "arima(1,0,0)"


def test_arima_predicts_through_the_per_sector_adapter():
    history = pd.DataFrame(
        {
            "cd_setor": ["A"] * 6,
            "CICLOS": [f"20260{n}" for n in range(1, 7)],
            "items": [100.0, 120.0, 110.0, 130.0, 125.0, 135.0],
            "opening_date": pd.to_datetime(
                ["2026-01-05", "2026-02-02", "2026-03-02", "2026-04-02", "2026-05-04", "2026-06-01"]
            ),
        }
    )
    targets = pd.DataFrame(
        {"cd_setor": ["A"], "CICLOS": ["202607"], "opening_date": pd.to_datetime(["2026-07-06"])}
    )
    candidate = REGISTRY.build(ArimaParams(order=(1, 0, 0)))

    predictions = candidate.fit_predict(history, targets)

    assert len(predictions) == 1
    assert predictions["items_pred"].iloc[0] > 0
