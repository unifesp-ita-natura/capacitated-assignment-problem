"""ETS candidate: exponential smoothing per sector, plugged in through the per-sector Adapter."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.config.schema import ETSParams
from src.forecasting.model import REGISTRY, ForecastCandidate, InsufficientHistoryError, per_sector

# Same contract as the ARIMA candidate: statsmodels declining a sector (a
# multiplicative component on a series with zeros, a singular fit) skips
# that sector; anything else propagates as a real bug.
_FIT_FAILURES = (np.linalg.LinAlgError, ValueError, IndexError)


def _minimum_observations(params: ETSParams) -> int:
    """Shortest history the requested components can be fitted on.

    Counts the smoothing parameters (alpha; a trend adds beta, damping adds
    phi) and asks for two observations each plus two, the same floor the
    ARIMA candidate puts on its coefficients: 4 cycles for ETS(A,N,N), 6
    with a trend, 8 damped. A seasonal model also needs two full seasons,
    the usual minimum to separate the seasonal pattern from the level.
    """
    parameters = 1 + (params.trend is not None) + params.damped_trend
    minimum = 2 * parameters + 2
    if params.seasonal is not None:
        minimum = max(minimum, 2 * params.seasonal_periods)
    return minimum


def _fit_and_forecast(params: ETSParams, series: pd.Series, horizon: int) -> pd.Series:
    """Fit ETS on one sector's series and forecast `horizon` cycles ahead."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    with warnings.catch_warnings():
        # Convergence chatter, one line per sector per fold.
        warnings.simplefilter("ignore")
        model = ETSModel(
            series.to_numpy(dtype=float),
            error=params.error,
            trend=params.trend,
            damped_trend=params.damped_trend,
            seasonal=params.seasonal,
            seasonal_periods=params.seasonal_periods,
        )
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=horizon)
    return pd.Series(np.asarray(forecast, dtype=float))


def _ets_fn(params: ETSParams) -> Callable[[pd.Series, int], pd.Series]:
    minimum = _minimum_observations(params)

    def predict(series: pd.Series, horizon: int) -> pd.Series:
        if len(series) < minimum:
            raise InsufficientHistoryError(
                f"{_candidate_name(params)} needs >= {minimum} cycles of history, got {len(series)}"
            )
        try:
            forecast = _fit_and_forecast(params, series, horizon)
        except _FIT_FAILURES as exc:
            raise InsufficientHistoryError(f"ETS could not fit this sector: {exc}") from exc
        if not np.isfinite(forecast).all():
            raise InsufficientHistoryError("ETS produced a non-finite forecast")
        # An additive trend extrapolates below zero on a declining sector.
        return forecast.clip(lower=0.0)

    return predict


def _candidate_name(params: ETSParams) -> str:
    """Hyndman's ETS(error,trend,seasonal) label, e.g. `ets(A,Ad,N)` or `ets(A,N,A,19)`."""
    letter = {"add": "A", "mul": "M", None: "N"}
    trend = letter[params.trend] + ("d" if params.damped_trend else "")
    name = f"ets({letter[params.error]},{trend},{letter[params.seasonal]}"
    if params.seasonal is not None:
        name += f",{params.seasonal_periods}"
    return name + ")"


_ALPHA_GRID = np.round(np.arange(0.01, 1.001, 0.01), 2)


def _smooth(matrix: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Run simple exponential smoothing down every row (sector) of a sectors x cycles matrix.

    A missing cycle (NaN) leaves the level untouched. The level starts at
    the sector's first observation, not its mean: a mean over the whole
    history already contains the cycles being "predicted" in-sample, which
    drags the pooled alpha to ~0 (measured in experiments/ets_level).
    Returns each sector's final level and the sum of absolute
    one-step-ahead errors over every cycle after the first.
    """
    level = np.full(matrix.shape[0], np.nan)
    error = 0.0
    for observed in matrix.T:
        seen = ~np.isnan(observed)
        both = seen & ~np.isnan(level)
        error += np.abs(observed[both] - level[both]).sum()
        level[both] += alpha * (observed[both] - level[both])
        fresh = seen & ~both
        level[fresh] = observed[fresh]
    return level, error


class _SharedAlphaCandidate:
    """ETS(A,N,N) with one alpha for every sector, run on the whole panel at once.

    Fitting alpha per sector from 6-11 cycles mostly fits noise (see
    experiments/ets_level). A single alpha is estimated from every sector's
    history together, so it is far more stable.
    """

    def __init__(self, alpha: float | str) -> None:
        self.name = f"ets(A,N,N) alpha={alpha}"
        self._alpha = alpha
        self.chosen_alpha: float | None = None

    def fit_predict(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        matrix = history.pivot_table(
            index="cd_setor", columns="opening_date", values="items", aggfunc="sum"
        ).sort_index(axis=1)
        values = matrix.to_numpy(dtype=float)
        if self._alpha == "pooled":
            # ponytail: grid of 100 alphas; a scalar optimiser only if the
            # optimum turns out to sit between grid points in a way that matters.
            errors = [_smooth(values, alpha)[1] for alpha in _ALPHA_GRID]
            self.chosen_alpha = float(_ALPHA_GRID[int(np.argmin(errors))])
        else:
            self.chosen_alpha = float(self._alpha)
        levels = pd.Series(_smooth(values, self.chosen_alpha)[0], index=matrix.index)

        rows = targets[targets["cd_setor"].isin(levels.index)]
        return pd.DataFrame(
            {
                "cd_setor": rows["cd_setor"].to_numpy(),
                "CICLOS": rows["CICLOS"].to_numpy(),
                "items_pred": np.clip(rows["cd_setor"].map(levels).to_numpy(), 0, None),
            }
        )


@REGISTRY.register("ets")
def build_ets(params: ETSParams) -> ForecastCandidate:
    """Build the ETS candidate the config's components select.

    The components come from the config and apply to every sector, rather
    than being chosen per sector by AIC as the spec's section 4.4 describes.
    The reason is the same as for ARIMA: with 6-11 training cycles, a
    per-sector search selects noise. To compare model forms, list several
    `ets` entries in one experiment instead. Setting `alpha` swaps in the
    pooled ETS(A,N,N), which shares one smoothing weight across sectors.
    """
    if params.alpha is not None:
        return _SharedAlphaCandidate(params.alpha)
    return per_sector(name=_candidate_name(params), fn=_ets_fn(params))
