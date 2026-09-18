"""Error metrics the evaluation harness computes and scoreboard helpers for ranking strategies."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

# The metric evaluation.py ranks candidates by. Decided once, here, rather
# than left for each candidate to choose — see docs/agent-log for why MAE
# over RMSE/MASE: same unit as the forecast (items), easiest to explain
# to a non-technical audience. RMSE and MASE are still always computed and
# reported alongside it, just not used to rank.
PRIMARY_METRIC = "mae"


def mae(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    """Mean absolute error, in the same unit as `actual` (items)."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    """Root mean squared error; penalizes large misses more than `mae`."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mase(actual: npt.ArrayLike, predicted: npt.ArrayLike, in_sample_naive_mae: float) -> float:
    """Mean absolute scaled error: `mae` divided by a naive one-step benchmark's in-sample MAE.

    `in_sample_naive_mae` must be computed on the training data alone (the
    naive model's own one-cycle-ahead absolute errors, averaged), never on
    the holdout being scored — otherwise the scale itself leaks future
    data. Undefined (returns `inf`) when the benchmark had zero in-sample
    error, since there is nothing to scale against.
    """
    if in_sample_naive_mae == 0:
        return float("inf")
    return mae(actual, predicted) / in_sample_naive_mae


def mae_wmape(comparison: pd.DataFrame, forecast_col: str, actual_col: str) -> tuple[float, float]:
    """Mean absolute error and weighted MAPE between a forecast and actual column."""
    error = comparison[forecast_col] - comparison[actual_col]
    mae_val = float(error.abs().mean())
    wmape_val = float(error.abs().sum() / comparison[actual_col].sum())
    return mae_val, wmape_val


def build_scoreboard(scores: list[dict], sort_col: str) -> pd.DataFrame:
    """Assemble per-strategy score rows into a scoreboard sorted best-first."""
    return pd.DataFrame(scores).set_index("strategy").sort_values(sort_col)


def select_best(scoreboard: pd.DataFrame, sort_col: str) -> str:
    """Name of the strategy with the lowest value in `sort_col`."""
    return str(scoreboard[sort_col].idxmin())
