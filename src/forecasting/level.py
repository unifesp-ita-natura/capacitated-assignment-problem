"""Forecast each sector's next-campanha order volume from its historical campanha totals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

CAMPANHA_TOTAL_COLUMNS = ["campanha_id", "sector", "campanha_total"]

LevelStrategy = Callable[[pd.DataFrame, pd.DataFrame], float]


def build_campanha_totals(orders: pd.DataFrame) -> pd.DataFrame:
    """Each sector's total orders per campanha, from a raw orders table."""
    return (
        orders.groupby(["campanha_id", "sector"], as_index=False)["orders"]
        .sum()
        .rename(columns={"orders": "campanha_total"})
    )


def extrapolate_linear_trend(values: np.ndarray) -> float:
    """Extrapolate one non-negative value from a short linear trend fit on `values`."""
    x_values = np.arange(len(values), dtype=float)
    slope = np.polyfit(x_values, values, deg=1)[0] if len(values) > 1 else 0.0
    return max(0.0, float(values[-1] + slope))


def level_naive_last(campanha_totals: pd.DataFrame, campanhas: pd.DataFrame) -> float:
    """Forecast the next campanha total as simply the last observed campanha total."""
    return float(campanha_totals.sort_values("campanha_id")["campanha_total"].iloc[-1])


def level_linear_trend(campanha_totals: pd.DataFrame, campanhas: pd.DataFrame) -> float:
    """Forecast the next campanha total from a short linear trend (the pipeline's default)."""
    values = campanha_totals.sort_values("campanha_id")["campanha_total"].to_numpy(dtype=float)
    return extrapolate_linear_trend(values)


def level_holt_ets(campanha_totals: pd.DataFrame, campanhas: pd.DataFrame) -> float:
    """Forecast the next campanha total with Holt's linear-trend exponential smoothing."""
    from statsmodels.tsa.holtwinters import Holt

    values = campanha_totals.sort_values("campanha_id")["campanha_total"].to_numpy(dtype=float)
    fit = Holt(values, initialization_method="estimated").fit()
    return max(0.0, float(fit.forecast(1)[0]))


def level_arima(campanha_totals: pd.DataFrame, campanhas: pd.DataFrame) -> float:
    """Forecast the next campanha total with a small ARIMA(1,1,0) model."""
    from statsmodels.tsa.arima.model import ARIMA

    values = campanha_totals.sort_values("campanha_id")["campanha_total"].to_numpy(dtype=float)
    fit = ARIMA(values, order=(1, 1, 0)).fit()
    return max(0.0, float(fit.forecast(1)[0]))


def _build_calendar_regression_design(merged: pd.DataFrame) -> pd.DataFrame:
    """OLS design matrix (const, campanha_id, calendar month) for calendar regression."""
    import statsmodels.api as sm

    return sm.add_constant(
        pd.DataFrame(
            {
                "campanha_id": merged["campanha_id"].astype(float),
                "month": merged["open_date"].dt.month.astype(float),
            }
        ),
        has_constant="add",
    )


def level_calendar_regression(campanha_totals: pd.DataFrame, campanhas: pd.DataFrame) -> float:
    """Forecast the next campanha total via OLS on campanha index + calendar month."""
    import statsmodels.api as sm

    merged = campanha_totals.merge(campanhas[["campanha_id", "open_date"]], on="campanha_id")
    merged = merged.sort_values("campanha_id")
    design = _build_calendar_regression_design(merged)
    model = sm.OLS(merged["campanha_total"].astype(float), design).fit()

    next_campanha_id = merged["campanha_id"].max() + 1
    next_campanha = campanhas.loc[campanhas["campanha_id"].eq(next_campanha_id)].iloc[0]
    next_design = pd.DataFrame(
        {
            "const": [1.0],
            "campanha_id": [float(next_campanha_id)],
            "month": [float(next_campanha["open_date"].month)],
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


def forecast_campanha_total_with(
    strategy: LevelStrategy,
    historical_campanha_totals: pd.DataFrame,
    campanhas: pd.DataFrame,
) -> pd.DataFrame:
    """Apply one level-forecast strategy to every sector's historical campanha totals."""
    return (
        historical_campanha_totals.groupby("sector")
        .apply(lambda group: strategy(group, campanhas), include_groups=False)
        .rename("forecast_campanha_total")
        .reset_index()
    )


def score_level_strategy(
    forecast: pd.DataFrame, actual_campanha_total: pd.DataFrame
) -> tuple[float, float]:
    """Score one level-forecast strategy's per-sector totals against the actual campanha."""
    comparison = forecast.merge(actual_campanha_total, on="sector")
    error = comparison["forecast_campanha_total"] - comparison["actual_campanha_total"]
    mae = float(error.abs().mean())
    wmape = float(error.abs().sum() / comparison["actual_campanha_total"].sum())
    return mae, wmape


def score_level_strategies(
    strategies: dict[str, LevelStrategy],
    historical_campanha_totals: pd.DataFrame,
    campanhas: pd.DataFrame,
    actual_campanha_total: pd.DataFrame,
) -> pd.DataFrame:
    """Forecast and score every level strategy against the held-out campanha, sorted by WMAPE."""
    scores = []
    for name, strategy in strategies.items():
        forecast = forecast_campanha_total_with(strategy, historical_campanha_totals, campanhas)
        mae, wmape = score_level_strategy(forecast, actual_campanha_total)
        scores.append({"strategy": name, "campanha_total_MAE": mae, "campanha_total_WMAPE": wmape})
    return pd.DataFrame(scores).set_index("strategy").sort_values("campanha_total_WMAPE")


def select_best_level_strategy(level_scoreboard: pd.DataFrame) -> str:
    """Name of the level strategy with the lowest WMAPE on the held-out campanha."""
    return str(level_scoreboard["campanha_total_WMAPE"].idxmin())
