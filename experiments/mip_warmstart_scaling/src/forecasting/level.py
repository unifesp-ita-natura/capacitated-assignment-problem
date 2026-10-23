"""Forecast each sector's next-cycle order volume from its historical cycle totals."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

LevelStrategy = Callable[[pd.DataFrame, pd.DataFrame, int], float]


def extrapolate_linear_trend(values: np.ndarray, step: int = 1) -> float:
    """Extrapolate one non-negative value `step` cycles out from a short linear trend fit."""
    x_values = np.arange(len(values), dtype=float)
    slope = np.polyfit(x_values, values, deg=1)[0] if len(values) > 1 else 0.0
    return max(0.0, float(values[-1] + slope * step))


def level_naive_last(cycle_totals: pd.DataFrame, cycles: pd.DataFrame, step: int = 1) -> float:
    """Forecast a future cycle's total as simply the last observed cycle total."""
    return float(cycle_totals.sort_values("cycle_id")["cycle_total"].iloc[-1])


def level_linear_trend(cycle_totals: pd.DataFrame, cycles: pd.DataFrame, step: int = 1) -> float:
    """Forecast a future cycle's total `step` cycles out from a short linear trend."""
    values = cycle_totals.sort_values("cycle_id")["cycle_total"].to_numpy(dtype=float)
    return extrapolate_linear_trend(values, step)


def level_holt_ets(cycle_totals: pd.DataFrame, cycles: pd.DataFrame, step: int = 1) -> float:
    """Forecast a future cycle's total `step` cycles out with Holt's linear-trend smoothing."""
    from statsmodels.tsa.holtwinters import Holt

    values = cycle_totals.sort_values("cycle_id")["cycle_total"].to_numpy(dtype=float)
    fit = Holt(values, initialization_method="estimated").fit()
    return max(0.0, float(fit.forecast(step)[-1]))


def level_arima(cycle_totals: pd.DataFrame, cycles: pd.DataFrame, step: int = 1) -> float:
    """Forecast a future cycle's total `step` cycles out with a small ARIMA(1,1,0) model."""
    from statsmodels.tsa.arima.model import ARIMA

    values = cycle_totals.sort_values("cycle_id")["cycle_total"].to_numpy(dtype=float)
    fit = ARIMA(values, order=(1, 1, 0)).fit()
    return max(0.0, float(fit.forecast(step)[-1]))


def _build_calendar_regression_design(merged: pd.DataFrame) -> pd.DataFrame:
    """OLS design matrix (const, cycle_id, calendar month) for calendar regression."""
    import statsmodels.api as sm

    return sm.add_constant(
        pd.DataFrame(
            {
                "cycle_id": merged["cycle_id"].astype(float),
                "month": merged["open_date"].dt.month.astype(float),
            }
        ),
        has_constant="add",
    )


def level_calendar_regression(
    cycle_totals: pd.DataFrame, cycles: pd.DataFrame, step: int = 1
) -> float:
    """Forecast a future cycle's total `step` cycles out via OLS on cycle index + month."""
    import statsmodels.api as sm

    merged = cycle_totals.merge(cycles[["cycle_id", "open_date"]], on="cycle_id")
    merged = merged.sort_values("cycle_id")
    design = _build_calendar_regression_design(merged)
    model = sm.OLS(merged["cycle_total"].astype(float), design).fit()

    next_cycle_id = merged["cycle_id"].max() + step
    next_cycle = cycles.loc[cycles["cycle_id"].eq(next_cycle_id)].iloc[0]
    next_design = pd.DataFrame(
        {
            "const": [1.0],
            "cycle_id": [float(next_cycle_id)],
            "month": [float(next_cycle["open_date"].month)],
        }
    )
    return max(0.0, float(model.predict(next_design)[0]))


LEVEL_STRATEGIES: dict[str, LevelStrategy] = {
    "naive_last": level_naive_last,
    "linear_trend": level_linear_trend,
    "holt_ets": level_holt_ets,
    "arima": level_arima,
    "calendar_regression": level_calendar_regression,
}


def forecast_cycle_total_with(
    strategy: LevelStrategy,
    historical_cycle_totals: pd.DataFrame,
    cycles: pd.DataFrame,
    steps: Iterable[int] = (1,),
) -> pd.DataFrame:
    """Apply one level-forecast strategy to every sector's historical cycle totals,
    once per `step` (cycles beyond the last historical cycle_id), returning a
    `[sector, cycle_id, forecast_cycle_total]` frame with one row per (sector, step)."""
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
