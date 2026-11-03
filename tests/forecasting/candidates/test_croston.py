"""Tests for the Croston / SBA / TSB intermittent-demand candidate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.forecasting.candidates  # noqa: F401 - registers candidates into REGISTRY
from src.config.schema import CrostonParams
from src.forecasting.candidates.croston import _croston_classic, _croston_fn, _croston_sba
from src.forecasting.model import REGISTRY, InsufficientHistoryError


def test_classic_croston_is_size_over_interval_for_a_regular_series():
    # Demand of 10 every 2nd cycle: size 10, interval 2 -> 5 per cycle.
    assert _croston_classic(np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0]), alpha=0.1) == (
        pytest.approx(5.0)
    )


def test_sba_shrinks_classic_by_the_bias_correction():
    values = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0])

    assert _croston_sba(values, 0.2) == pytest.approx(0.9 * _croston_classic(values, 0.2))


@pytest.mark.parametrize("variant", ["classic", "sba", "tsb"])
def test_all_variants_forecast_a_flat_non_negative_level(variant):
    predict = _croston_fn(CrostonParams(variant=variant))

    forecast = predict(pd.Series([0.0, 4.0, 0.0, 0.0, 6.0, 5.0]), 3)

    assert len(forecast) == 3
    assert forecast.nunique() == 1
    assert (forecast >= 0).all()


def test_an_all_zero_series_forecasts_zero():
    forecast = _croston_fn(CrostonParams(variant="tsb"))(pd.Series([0.0, 0.0, 0.0, 0.0]), 2)

    assert (forecast == 0).all()


def test_tsb_decays_when_demand_stops_but_classic_does_not():
    series = pd.Series([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    classic = _croston_fn(CrostonParams(variant="classic"))(series, 1).iloc[0]
    tsb = _croston_fn(CrostonParams(variant="tsb", beta=0.3))(series, 1).iloc[0]

    assert tsb < classic


def test_short_series_is_skipped_rather_than_fitted():
    with pytest.raises(InsufficientHistoryError, match="cycles of history"):
        _croston_fn(CrostonParams())(pd.Series([1.0, 0.0]), 1)


def test_registry_builds_a_named_candidate():
    candidate = REGISTRY.build(CrostonParams(variant="tsb"))

    assert candidate.name == "croston:tsb"
