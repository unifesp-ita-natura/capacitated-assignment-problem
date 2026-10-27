"""Intermittent-demand candidate (Croston / SBA / TSB), through the per-sector Adapter."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.config.schema import CrostonParams
from src.forecasting.model import (
    REGISTRY,
    ForecastCandidate,
    InsufficientHistoryError,
    per_sector,
)

_MINIMUM_OBSERVATIONS = 3


def _croston_classic(values: np.ndarray, alpha: float) -> float:
    """Classic Croston: smooth demand size and inter-demand interval, forecast size/interval."""
    nonzero = np.flatnonzero(values > 0)
    if nonzero.size == 0:
        return 0.0
    size = float(values[nonzero[0]])
    interval = float(nonzero[0] + 1)
    last = nonzero[0]
    for idx in nonzero[1:]:
        size += alpha * (values[idx] - size)
        interval += alpha * ((idx - last) - interval)
        last = idx
    return size / interval if interval > 0 else size


def _croston_sba(values: np.ndarray, alpha: float) -> float:
    """Syntetos-Boylan Approximation: Croston scaled by (1 - alpha/2) to debias the ratio."""
    return (1.0 - alpha / 2.0) * _croston_classic(values, alpha)


def _croston_tsb(values: np.ndarray, alpha: float, beta: float) -> float:
    """Teunter-Syntetos-Babai: smooth the demand probability, so quiet series decay to zero."""
    nonzero = np.flatnonzero(values > 0)
    if nonzero.size == 0:
        return 0.0
    size = float(values[nonzero[0]])
    prob = 1.0 / (nonzero[0] + 1)
    for value in values[nonzero[0] + 1 :]:
        occurred = value > 0
        prob += beta * (float(occurred) - prob)
        if occurred:
            size += alpha * (value - size)
    return prob * size


def _croston_fn(params: CrostonParams) -> Callable[[pd.Series, int], pd.Series]:
    def predict(series: pd.Series, horizon: int) -> pd.Series:
        if len(series) < _MINIMUM_OBSERVATIONS:
            raise InsufficientHistoryError(
                f"croston needs >= {_MINIMUM_OBSERVATIONS} cycles of history, got {len(series)}"
            )
        values = series.to_numpy(dtype=float)
        if params.variant == "classic":
            level = _croston_classic(values, params.alpha)
        elif params.variant == "sba":
            level = _croston_sba(values, params.alpha)
        else:
            level = _croston_tsb(values, params.alpha, params.beta)
        level = max(level, 0.0)
        return pd.Series([level] * horizon)

    return predict


@REGISTRY.register("croston")
def build_croston(params: CrostonParams) -> ForecastCandidate:
    """Build the intermittent-demand candidate, scaled and opening-adjusted by the Adapter."""
    return per_sector(name=f"croston:{params.variant}", fn=_croston_fn(params))
