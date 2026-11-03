"""Prophet candidate: fitted per (sector, city) series, through the per-sector Adapter."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.config.schema import ProphetParams
from src.forecasting.model import (
    REGISTRY,
    ForecastCandidate,
    InsufficientHistoryError,
    per_sector,
)

_MINIMUM_OBSERVATIONS = 6
_FIT_FAILURES = (np.linalg.LinAlgError, ValueError, RuntimeError, IndexError)


def _silence_prophet_logging() -> None:
    for name in ("prophet", "cmdstanpy"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.handlers.clear()
        logger.propagate = False
        logger.disabled = True


_silence_prophet_logging()


def _future_dates(history_dates: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """The opening dates of the next `horizon` cycles, stepped by the median observed gap."""
    ordered = history_dates.sort_values()
    step = ordered.to_series().diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(days=21)
    last = ordered[-1]
    return pd.DatetimeIndex([last + step * (i + 1) for i in range(horizon)])


def _fit_and_forecast(params: ProphetParams, series: pd.Series, horizon: int) -> pd.Series:
    """Fit Prophet on one series' (opening_date, items) and forecast `horizon` cycles."""
    from prophet import Prophet

    _silence_prophet_logging()
    train = pd.DataFrame({"ds": series.index, "y": series.to_numpy(dtype=float)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(
            growth=params.growth,
            n_changepoints=min(params.n_changepoints, max(len(series) - 1, 0)),
            changepoint_prior_scale=params.changepoint_prior_scale,
            seasonality_mode=params.seasonality_mode,
            yearly_seasonality=params.yearly_seasonality,
            weekly_seasonality=params.weekly_seasonality,
            daily_seasonality=params.daily_seasonality,
        )
        model.fit(train)
        future = pd.DataFrame({"ds": _future_dates(pd.DatetimeIndex(series.index), horizon)})
        forecast = model.predict(future)["yhat"]
    return pd.Series(np.asarray(forecast, dtype=float))


def _prophet_fn(params: ProphetParams) -> Callable[[pd.Series, int], pd.Series]:
    def predict(series: pd.Series, horizon: int) -> pd.Series:
        if len(series) < _MINIMUM_OBSERVATIONS:
            raise InsufficientHistoryError(
                f"prophet needs >= {_MINIMUM_OBSERVATIONS} cycles of history, got {len(series)}"
            )
        try:
            forecast = _fit_and_forecast(params, series, horizon)
        except _FIT_FAILURES as exc:
            raise InsufficientHistoryError(f"prophet could not fit this series: {exc}") from exc
        if not np.isfinite(forecast).all():
            raise InsufficientHistoryError("prophet produced a non-finite forecast")
        return forecast.clip(lower=0.0)

    return predict


@REGISTRY.register("prophet")
def build_prophet(params: ProphetParams) -> ForecastCandidate:
    """Build the per-series Prophet candidate, scaled and opening-adjusted by the Adapter."""
    return per_sector(name="prophet", fn=_prophet_fn(params))
