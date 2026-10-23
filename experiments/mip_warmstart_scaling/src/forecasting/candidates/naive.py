"""Naive forecasting candidate: the mandatory baseline every other technique must beat."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.config.schema import NaiveParams
from src.forecasting.model import REGISTRY, ForecastCandidate, InsufficientHistoryError, per_sector


def _last_value(series: pd.Series, horizon: int) -> pd.Series:
    if series.empty:
        raise InsufficientHistoryError("no history to repeat the last value from")
    return pd.Series([series.iloc[-1]] * horizon)


def _mean(series: pd.Series, horizon: int) -> pd.Series:
    if series.empty:
        raise InsufficientHistoryError("no history to average")
    return pd.Series([series.mean()] * horizon)


def _seasonal_naive_fn(season_length: int) -> Callable[[pd.Series, int], pd.Series]:
    def _seasonal_naive(series: pd.Series, horizon: int) -> pd.Series:
        if len(series) < season_length:
            raise InsufficientHistoryError(
                f"need >= {season_length} cycles of history for seasonal_naive, got {len(series)} "
                "(this base currently spans one year, so seasonal_naive has no prior cycle to "
                "repeat from yet — see docs/agent-log)"
            )
        last_season = list(series.iloc[-season_length:])
        repeats = -(-horizon // season_length)  # ceil division
        return pd.Series((last_season * repeats)[:horizon])

    return _seasonal_naive


def _strategy_fn(params: NaiveParams) -> Callable[[pd.Series, int], pd.Series]:
    if params.strategy == "last_value":
        return _last_value
    if params.strategy == "mean":
        return _mean
    return _seasonal_naive_fn(params.season_length)


@REGISTRY.register("naive")
def build_naive(params: NaiveParams) -> ForecastCandidate:
    """Build the naive candidate the config's `strategy` selects."""
    return per_sector(name=f"naive:{params.strategy}", fn=_strategy_fn(params))
