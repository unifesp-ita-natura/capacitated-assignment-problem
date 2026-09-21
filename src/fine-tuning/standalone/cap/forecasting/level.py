"""Forecast each sector's next-cycle order volume from its historical cycle totals."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

LevelStrategy = Callable[[pd.DataFrame, pd.DataFrame, int], float]


def extrapolate_linear_trend(values: np.ndarray, step: int = 1) -> float:
    x_values = np.arange(len(values), dtype=float)
    slope = np.polyfit(x_values, values, deg=1)[0] if len(values) > 1 else 0.0
    return max(0.0, float(values[-1] + slope * step))


def level_naive_last(cycle_totals, cycles, step: int = 1) -> float:
    return float(cycle_totals.sort_values("cycle_id")["cycle_total"].iloc[-1])


def level_linear_trend(cycle_totals, cycles, step: int = 1) -> float:
    values = cycle_totals.sort_values("cycle_id")["cycle_total"].to_numpy(dtype=float)
    return extrapolate_linear_trend(values, step)


def level_holt_ets(cycle_totals, cycles, step: int = 1) -> float:
    from statsmodels.tsa.holtwinters import Holt  # optional extra

    values = cycle_totals.sort_values("cycle_id")["cycle_total"].to_numpy(dtype=float)
    fit = Holt(values, initialization_method="estimated").fit()
    return max(0.0, float(fit.forecast(step)[-1]))


def level_arima(cycle_totals, cycles, step: int = 1) -> float:
    from statsmodels.tsa.arima.model import ARIMA  # optional extra

    values = cycle_totals.sort_values("cycle_id")["cycle_total"].to_numpy(dtype=float)
    fit = ARIMA(values, order=(1, 1, 0)).fit()
    return max(0.0, float(fit.forecast(step)[-1]))


LEVEL_STRATEGIES: dict[str, LevelStrategy] = {
    "naive_last": level_naive_last,
    "linear_trend": level_linear_trend,
    "holt_ets": level_holt_ets,
    "arima": level_arima,
}


def forecast_cycle_total_with(
    strategy, historical_cycle_totals, cycles, steps: Iterable[int] = (1,)
):
    base_cycle_id = int(historical_cycle_totals["cycle_id"].max())
    forecasts = []
    for step in steps:
        forecast = (
            historical_cycle_totals.groupby("sector")
            .apply(lambda group: strategy(group, cycles, step), include_groups=False)
            .rename("forecast_cycle_total")
            .reset_index()
        )
        forecast["cycle_id"] = base_cycle_id + step
        forecasts.append(forecast)
    return pd.concat(forecasts, ignore_index=True)[["sector", "cycle_id", "forecast_cycle_total"]]
