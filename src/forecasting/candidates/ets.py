"""ETS candidate: exponential smoothing per sector, plugged in through the per-sector Adapter."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.config.schema import ETSParams
from src.forecasting.model import (
    QUERY_KEYS,
    REGISTRY,
    ForecastCandidate,
    InsufficientHistoryError,
    opening_adjusted,
    per_sector,
)

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
_EXPONENT_GRID = np.round(np.arange(0.0, 1.001, 0.1), 1)


def _smooth(
    matrix: np.ndarray, alpha: float, scale: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Run simple exponential smoothing down every row (sector) of a sectors x cycles matrix.

    A missing cycle (NaN) leaves the level untouched. The level starts at
    the sector's first observation, not its mean: a mean over the whole
    history already contains the cycles being "predicted" in-sample, which
    drags the pooled alpha to ~0 (measured in experiments/ets_level).
    With `scale`, each cycle is divided by its own scale before smoothing
    and the one-step forecast is multiplied back, so the errors stay in
    items. Returns each sector's final level (per unit of scale) and the
    sum of absolute one-step-ahead errors over every cycle after the first.
    """
    scale = np.ones_like(matrix) if scale is None else scale
    level = np.full(matrix.shape[0], np.nan)
    error = 0.0
    for observed, factor in zip((matrix / scale).T, scale.T, strict=True):
        seen = ~np.isnan(observed)
        both = seen & ~np.isnan(level)
        error += (np.abs(observed[both] - level[both]) * factor[both]).sum()
        level[both] += alpha * (observed[both] - level[both])
        fresh = seen & ~both
        level[fresh] = observed[fresh]
    return level, error


def _grid(value: float | str, pooled_grid: np.ndarray) -> np.ndarray:
    return pooled_grid if value == "pooled" else np.array([float(value)])


class _SharedAlphaCandidate:
    """ETS(A,N,N) with one alpha for every sector, run on the whole panel at once.

    Fitting alpha per sector from 6-11 cycles mostly fits noise (see
    experiments/ets_level). A single alpha is estimated from every sector's
    history together, so it is far more stable. A cycle's items are taken
    to scale with its length as cycle_days ** exponent, the exponent picked
    with alpha unless the config fixes it: the level is smoothed per unit of
    that scale, and each scenario query gets the level times its own window
    length to the exponent. Its own estimate rather than `window_scaled`'s,
    because here the exponent can be chosen by forecast error.
    """

    def __init__(self, alpha: float | str, exponent: float | str = "pooled") -> None:
        self.name = f"ets(A,N,N) alpha={alpha}"
        if exponent != "pooled":
            self.name += f" days^{exponent}"
        self._alpha = alpha
        self._exponent = exponent
        self.chosen_alpha: float | None = None
        self.chosen_exponent: float | None = None

    def fit_predict(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        items = _by_sector_and_cycle(history, "items")
        days = _by_sector_and_cycle(history, "cycle_days").reindex_like(items)
        levels = pd.Series(self._fit(items.to_numpy(float), days.to_numpy(float)), items.index)

        rows = targets[targets["cd_setor"].isin(levels.index)]
        predicted = rows["cd_setor"].map(levels).to_numpy()
        predicted = predicted * rows["cycle_days"].to_numpy(float) ** self.chosen_exponent
        return (
            rows[QUERY_KEYS].assign(items_pred=np.clip(predicted, 0, None)).reset_index(drop=True)
        )

    def _fit(self, items: np.ndarray, days: np.ndarray) -> np.ndarray:
        """Pick alpha and the exponent with the lowest in-sample error; return the final levels."""
        # ponytail: exhaustive grid, 100 alphas x 11 exponents; a scalar
        # optimiser only if the optimum sits between grid points in a way
        # that matters.
        settings = [
            (a, e)
            for a in _grid(self._alpha, _ALPHA_GRID)
            for e in _grid(self._exponent, _EXPONENT_GRID)
        ]
        errors = [_smooth(items, a, days**e)[1] for a, e in settings]
        self.chosen_alpha, self.chosen_exponent = map(float, settings[int(np.argmin(errors))])
        return _smooth(items, self.chosen_alpha, days**self.chosen_exponent)[0]


def _by_sector_and_cycle(history: pd.DataFrame, column: str) -> pd.DataFrame:
    return history.pivot_table(
        index="cd_setor", columns="opening_date", values=column, aggfunc="sum"
    ).sort_index(axis=1)


@REGISTRY.register("ets")
def build_ets(params: ETSParams) -> ForecastCandidate:
    """Build the ETS candidate the config's components select.

    The components come from the config and apply to every sector, rather
    than being chosen per sector by AIC as the spec's section 4.4 describes.
    The reason is the same as for ARIMA: with 6-11 training cycles, a
    per-sector search selects noise. To compare model forms, list several
    `ets` entries in one experiment instead. Setting `alpha` swaps in the
    pooled ETS(A,N,N), which shares one smoothing weight across sectors.
    Either way the forecast is scaled to each scenario's window length and
    adjusted for the day it opens (`model.opening_adjusted`).
    """
    if params.alpha is not None:
        return opening_adjusted(_SharedAlphaCandidate(params.alpha, params.cycle_days_exponent))
    return per_sector(name=_candidate_name(params), fn=_ets_fn(params))
