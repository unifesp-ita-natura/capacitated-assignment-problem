"""Tests for the ETS candidate and its history guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import src.forecasting.candidates  # noqa: F401 - registers candidates into REGISTRY
from src.config.schema import ETSParams
from src.forecasting.candidates.ets import _candidate_name, _ets_fn, _smooth
from src.forecasting.model import REGISTRY, InsufficientHistoryError


def test_simple_smoothing_forecasts_inside_the_range_of_the_history():
    # ETS(A,N,N) is a weighted mean of the past, so it can't leave the range.
    series = pd.Series([100.0, 140.0, 90.0, 130.0, 110.0, 120.0, 105.0])

    forecast = _ets_fn(ETSParams())(series, 2)

    assert len(forecast) == 2
    assert series.min() <= forecast.iloc[0] <= series.max()


def test_short_series_is_skipped_rather_than_fitted():
    predict = _ets_fn(ETSParams(trend="add", damped_trend=True))

    with pytest.raises(InsufficientHistoryError, match="cycles of history"):
        predict(pd.Series([100.0, 110.0, 120.0, 130.0]), 1)


def test_forecast_is_never_negative():
    predict = _ets_fn(ETSParams(trend="add"))

    forecast = predict(pd.Series([1000.0, 800.0, 600.0, 400.0, 200.0, 100.0, 50.0]), 5)

    assert (forecast >= 0).all()


def test_multiplicative_error_on_a_series_with_zeros_skips_the_sector():
    predict = _ets_fn(ETSParams(error="mul"))

    with pytest.raises(InsufficientHistoryError, match="could not fit"):
        predict(pd.Series([100.0, 0.0, 120.0, 130.0, 0.0, 110.0, 90.0]), 1)


def test_damping_without_a_trend_is_rejected_by_the_config():
    with pytest.raises(ValidationError, match="damped_trend"):
        ETSParams(damped_trend=True)


def test_candidate_is_named_after_its_components():
    assert _candidate_name(ETSParams()) == "ets(A,N,N)"
    assert _candidate_name(ETSParams(trend="add", damped_trend=True)) == "ets(A,Ad,N)"
    seasonal = ETSParams(error="mul", seasonal="mul", seasonal_periods=19)
    assert _candidate_name(seasonal) == "ets(M,N,M,19)"


def test_registry_builds_the_ets_candidate_from_config():
    assert REGISTRY.build(ETSParams()).name == "ets(A,N,N)"


def _panel(
    series_by_sector: dict[str, list[float]], days: dict[str, list[int]] | None = None
) -> pd.DataFrame:
    days = days or {}
    return pd.DataFrame(
        [
            {
                "cd_setor": sector,
                "CICLOS": f"2026{n:02d}",
                "items": value,
                "opening_date": pd.Timestamp("2026-01-05") + pd.Timedelta(days=21 * n),
                "cycle_days": days.get(sector, [21] * len(values))[n],
            }
            for sector, values in series_by_sector.items()
            for n, value in enumerate(values)
        ]
    ).assign(window_start=lambda frame: frame["opening_date"])


def _next_cycle(sectors: list[str], days: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cd_setor": sectors,
            "CICLOS": "202699",
            "window_start": pd.Timestamp("2026-12-01"),
            "opening_date": pd.Timestamp("2026-12-01"),
            "cycle_days": days or [21] * len(sectors),
        }
    )


def test_smoothing_by_hand():
    # level 100 -> 100 + 0.5 * (200 - 100) = 150, one-step error |200 - 100| = 100
    level, error = _smooth(np.array([[100.0, 200.0]]), 0.5)

    assert level[0] == 150.0
    assert error == 100.0


def test_a_missing_cycle_leaves_the_level_untouched():
    level, _ = _smooth(np.array([[100.0, np.nan, 200.0]]), 0.5)

    assert level[0] == 150.0


def test_shared_alpha_forecasts_every_sector_with_its_own_level():
    candidate = REGISTRY.build(ETSParams(alpha=0.5))

    predictions = candidate.fit_predict(
        _panel({"A": [100.0, 200.0], "B": [10.0, 30.0]}), _next_cycle(["A", "B"])
    )

    by_sector = predictions.set_index("cd_setor")["items_pred"]
    assert by_sector.to_dict() == pytest.approx({"A": 150.0, "B": 20.0})


def test_pooled_alpha_is_high_when_demand_shifts_and_stays():
    # Every sector steps to a new level and stays: the last value is right.
    candidate = REGISTRY.build(ETSParams(alpha="pooled"))

    candidate.fit_predict(
        _panel({"A": [100.0] * 4 + [300.0] * 4, "B": [50.0] * 4 + [10.0] * 4}),
        _next_cycle(["A", "B"]),
    )

    assert candidate.chosen_alpha == 1.0


def test_pooled_alpha_is_low_when_demand_bounces_around_a_mean():
    candidate = REGISTRY.build(ETSParams(alpha="pooled"))

    candidate.fit_predict(
        _panel({"A": [100.0, 140.0, 60.0, 130.0, 70.0, 120.0, 80.0, 110.0]}), _next_cycle(["A"])
    )

    assert candidate.chosen_alpha < 0.5


def test_shared_alpha_is_rejected_on_a_model_with_a_trend():
    with pytest.raises(ValidationError, match="shared alpha"):
        ETSParams(trend="add", alpha=0.3)


def test_exponent_one_scales_the_forecast_to_the_queried_window():
    # 210 items in 21 days is 10 a day, so a 14-day window gets 140.
    candidate = REGISTRY.build(ETSParams(alpha=0.5, cycle_days_exponent=1.0))

    predictions = candidate.fit_predict(
        _panel({"A": [210.0, 210.0]}), _next_cycle(["A", "A"], [21, 14])
    )

    assert predictions["items_pred"].tolist() == pytest.approx([210.0, 140.0])
    assert {"window_start", "cycle_days"} <= set(predictions.columns)


def test_pooled_exponent_is_one_when_items_follow_the_window_length():
    days = [21, 14, 21, 14, 21, 14]
    candidate = REGISTRY.build(ETSParams(alpha="pooled", cycle_days_exponent="pooled"))

    candidate.fit_predict(_panel({"A": [10.0 * d for d in days]}, {"A": days}), _next_cycle(["A"]))

    assert candidate.chosen_exponent == 1.0


def test_pooled_exponent_is_zero_when_items_ignore_the_window_length():
    days = [21, 14, 21, 14, 21, 14]
    candidate = REGISTRY.build(ETSParams(alpha="pooled", cycle_days_exponent="pooled"))

    candidate.fit_predict(_panel({"A": [200.0] * 6}, {"A": days}), _next_cycle(["A"]))

    assert candidate.chosen_exponent == 0.0


def test_a_fixed_exponent_is_rejected_without_a_shared_alpha():
    with pytest.raises(ValidationError, match="needs a shared alpha"):
        ETSParams(cycle_days_exponent=1.0)
