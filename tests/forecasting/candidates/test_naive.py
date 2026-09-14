"""Tests for the naive baseline candidate — the required first Strategy plugged into the harness."""

from __future__ import annotations

import pandas as pd
import pytest

import src.forecasting.candidates  # noqa: F401 - registers "naive" into REGISTRY
from src.config.schema import NaiveParams
from src.forecasting.candidates.naive import _last_value, _mean, _seasonal_naive_fn
from src.forecasting.model import REGISTRY, InsufficientHistoryError

HISTORY = pd.DataFrame(
    {
        "cd_setor": ["A", "A", "A"],
        "CICLOS": ["1", "2", "3"],
        "items": [100, 200, 300],
        "opening_date": pd.to_datetime(["2026-01-01", "2026-01-22", "2026-02-12"]),
    }
)
TARGETS = pd.DataFrame(
    {"cd_setor": ["A"], "CICLOS": ["4"], "opening_date": pd.to_datetime(["2026-03-05"])}
)


def test_last_value_repeats_the_most_recent_observation():
    candidate = REGISTRY.build(NaiveParams(strategy="last_value"))

    predictions = candidate.fit_predict(HISTORY, TARGETS)

    assert predictions["items_pred"].iloc[0] == 300


def test_mean_predicts_the_historical_average():
    candidate = REGISTRY.build(NaiveParams(strategy="mean"))

    predictions = candidate.fit_predict(HISTORY, TARGETS)

    assert predictions["items_pred"].iloc[0] == 200  # mean(100, 200, 300)


def test_seasonal_naive_skips_a_sector_without_enough_seasons_yet():
    # Only 3 cycles of history exist anywhere in this base (see
    # docs/agent-log): a season_length of 4 can never be satisfied today,
    # and the candidate must skip rather than crash the whole run.
    candidate = REGISTRY.build(NaiveParams(strategy="seasonal_naive", season_length=4))

    predictions = candidate.fit_predict(HISTORY, TARGETS)

    assert predictions.empty


def test_naive_candidate_is_named_after_its_strategy():
    candidate = REGISTRY.build(NaiveParams(strategy="mean"))

    assert candidate.name == "naive:mean"


def test_naive_requires_at_least_one_historical_cycle():
    candidate = REGISTRY.build(NaiveParams(strategy="last_value"))
    empty_history = HISTORY.iloc[0:0]

    predictions = candidate.fit_predict(empty_history, TARGETS)

    assert predictions.empty


def test_naive_is_registered_by_default():
    assert "naive" in REGISTRY.known_models


# The per-sector Adapter only ever calls these with a non-empty series
# (pandas groupby never yields an empty group), so the empty-series guards
# below are unreachable through REGISTRY.build(...).fit_predict(...) and are
# tested directly instead — defensive checks for any future caller that
# invokes them outside the Adapter.


def test_last_value_rejects_an_empty_series_directly():
    with pytest.raises(InsufficientHistoryError, match="no history"):
        _last_value(pd.Series(dtype=float), horizon=1)


def test_mean_rejects_an_empty_series_directly():
    with pytest.raises(InsufficientHistoryError, match="no history"):
        _mean(pd.Series(dtype=float), horizon=1)


def test_seasonal_naive_tiles_the_last_season_across_a_longer_horizon():
    # 2 cycles of history, season_length=2, horizon=5 -> tile [a, b] to
    # cover 5 steps: exercises the ceil-division repeat count directly.
    series = pd.Series([10, 20])
    seasonal_naive = _seasonal_naive_fn(season_length=2)

    predicted = seasonal_naive(series, horizon=5)

    assert list(predicted) == [10, 20, 10, 20, 10]
