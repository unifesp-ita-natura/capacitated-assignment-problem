"""ARIMA/SARIMA candidate: one model per sector, plugged in through the per-sector Adapter."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.config.schema import ArimaParams
from src.forecasting.model import REGISTRY, ForecastCandidate, InsufficientHistoryError, per_sector

# statsmodels raises these when a series can't support the requested order, or
# when the optimizer can't converge on it. Neither is a bug in this repo — it's
# the model declining this particular sector — so they're translated into
# InsufficientHistoryError and the harness counts the sector as unpredicted.
# Anything outside this tuple propagates, so real bugs still surface.
_FIT_FAILURES = (np.linalg.LinAlgError, ValueError, IndexError)


def _minimum_observations(params: ArimaParams) -> int:
    """Shortest history that can support the requested order, plus a margin for estimation.

    An (p, d, q)(P, D, Q, m) model consumes d + D*m observations to
    difference, and estimates p + q + P + Q coefficients from what's left.
    Requiring two observations per coefficient is a low bar — it is *not* a
    claim that the estimate is any good at that length (see the experiment
    README on how thin this base's per-sector history is), only a floor
    below which the fit cannot be attempted at all.
    """
    p, d, q = params.order
    seasonal_p, seasonal_d, seasonal_q, season_length = params.seasonal_order or (0, 0, 0, 0)
    differencing = d + seasonal_d * season_length
    coefficients = p + q + seasonal_p + seasonal_q
    return differencing + 2 * coefficients + 2


def _fit_and_forecast(params: ArimaParams, series: pd.Series, horizon: int) -> pd.Series:
    """Fit SARIMAX on one sector's series and forecast `horizon` cycles ahead."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        # Convergence/frequency chatter, one line per sector per fold, would
        # drown the run's actual output at 600+ sectors.
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series.to_numpy(dtype=float),
            order=params.order,
            seasonal_order=params.seasonal_order or (0, 0, 0, 0),
            trend=params.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=horizon)
    return pd.Series(np.asarray(forecast, dtype=float))


def _arima_fn(params: ArimaParams) -> Callable[[pd.Series, int], pd.Series]:
    minimum = _minimum_observations(params)

    def predict(series: pd.Series, horizon: int) -> pd.Series:
        if len(series) < minimum:
            raise InsufficientHistoryError(
                f"{_candidate_name(params)} needs >= {minimum} cycles of history for order "
                f"{params.order} / seasonal {params.seasonal_order}, got {len(series)}"
            )
        try:
            forecast = _fit_and_forecast(params, series, horizon)
        except _FIT_FAILURES as exc:
            raise InsufficientHistoryError(f"SARIMAX could not fit this sector: {exc}") from exc
        if not np.isfinite(forecast).all():
            raise InsufficientHistoryError("SARIMAX produced a non-finite forecast")
        # Item demand can't be negative, and ForecastResult validates that
        # downstream (src/persistence/results.py) — an unconstrained linear
        # model can and does go below zero on low-volume sectors.
        return forecast.clip(lower=0.0)

    return predict


def _candidate_name(params: ArimaParams) -> str:
    """`sarima` when a seasonal order is set, `arima` when it isn't — they're the same model."""
    order = ",".join(str(term) for term in params.order)
    if params.seasonal_order is None:
        return f"arima({order})"
    seasonal = ",".join(str(term) for term in params.seasonal_order)
    return f"sarima({order})({seasonal})"


@REGISTRY.register("arima")
def build_arima(params: ArimaParams) -> ForecastCandidate:
    """Build the ARIMA/SARIMA candidate the config's order selects.

    The order is taken from the config and applied uniformly to every
    sector rather than searched per sector by AIC (which the spec's section
    4.4 calls for): with 6-11 training cycles per sector, a per-sector order
    search selects noise, and it multiplies runtime by the size of the grid.
    Revisit once the base spans more than one year.

    **Set `trend="c"` whenever `d` is 0.** `ArimaParams.trend` defaults to
    `None`, matching statsmodels, which means *no intercept* — the model is
    then forced to explain a series living around 4,000 items with the AR
    coefficient alone, which drives it to ~1 and degenerates the model into
    a random walk. Measured on the full base, `(1,0,0)` scores 2,619 items
    MAE without the intercept and 2,077 with it. With `d >= 1` the
    differencing already removes the level, and `trend="c"` becomes a drift
    term instead — a different modeling choice, not a fix.
    """
    return per_sector(name=_candidate_name(params), fn=_arima_fn(params))
