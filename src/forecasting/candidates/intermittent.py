"""Intermittent-demand candidates (TSB, ADIDA): per-series, for series with many zero cycles."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.config.schema import IntermittentParams
from src.forecasting.model import REGISTRY, ForecastCandidate, InsufficientHistoryError, per_sector


def tsb(values: np.ndarray, alpha_demand: float, alpha_probability: float) -> float:
    """Teunter-Syntetos-Babai: smoothed demand size times smoothed probability of any demand.

    The probability is updated every cycle, so a series that stops buying
    decays toward zero instead of holding its last size (Croston's flaw).
    Both start from the first observation that has demand.
    """
    bought = values > 0
    if not bought.any():
        return 0.0
    first = int(np.argmax(bought))
    size, probability = float(values[first]), 1.0
    for value in values[first + 1 :]:
        probability += alpha_probability * (float(value > 0) - probability)
        if value > 0:
            size += alpha_demand * (value - size)
    return size * probability


def adida(values: np.ndarray, alpha: float) -> float:
    """Aggregate-disaggregate: smooth bucket sums, then spread the forecast evenly per cycle.

    The bucket is the series' mean gap between cycles with demand, so each
    bucket usually holds one purchase and the smoothed series has few zeros.
    Buckets are cut from the most recent cycle backwards; a partial oldest
    bucket is dropped.
    """
    bought = np.flatnonzero(values > 0)
    if bought.size == 0:
        return 0.0
    bucket = max(1, round(len(values) / bought.size))
    usable = len(values) - len(values) % bucket
    sums = values[len(values) - usable :].reshape(-1, bucket).sum(axis=1)
    level = float(sums[0])
    for total in sums[1:]:
        level += alpha * (total - level)
    return level / bucket


def _strategy_fn(params: IntermittentParams) -> Callable[[pd.Series, int], pd.Series]:
    def predict(series: pd.Series, horizon: int) -> pd.Series:
        if series.empty:
            raise InsufficientHistoryError("no history to smooth")
        values = series.to_numpy(dtype=float)
        if params.strategy == "tsb":
            value = tsb(values, params.alpha_demand, params.alpha_probability)
        else:
            value = adida(values, params.alpha_demand)
        return pd.Series([value] * horizon)

    return predict


@REGISTRY.register("intermittent")
def build_intermittent(params: IntermittentParams) -> ForecastCandidate:
    """Build the TSB or ADIDA candidate; both forecast a flat level for every horizon step."""
    name = f"{params.strategy}:{params.alpha_demand}/{params.alpha_probability}"
    if params.strategy == "adida":
        name = f"adida:{params.alpha_demand}"
    return per_sector(name=name, fn=_strategy_fn(params))
