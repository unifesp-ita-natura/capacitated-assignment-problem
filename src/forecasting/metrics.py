"""Shared error-metric and scoreboard helpers for ranking forecast strategies."""

from __future__ import annotations

import pandas as pd


def mae_wmape(comparison: pd.DataFrame, forecast_col: str, actual_col: str) -> tuple[float, float]:
    """Mean absolute error and weighted MAPE between a forecast and actual column."""
    error = comparison[forecast_col] - comparison[actual_col]
    mae = float(error.abs().mean())
    wmape = float(error.abs().sum() / comparison[actual_col].sum())
    return mae, wmape


def build_scoreboard(scores: list[dict], sort_col: str) -> pd.DataFrame:
    """Assemble per-strategy score rows into a scoreboard sorted best-first."""
    return pd.DataFrame(scores).set_index("strategy").sort_values(sort_col)


def select_best(scoreboard: pd.DataFrame, sort_col: str) -> str:
    """Name of the strategy with the lowest value in `sort_col`."""
    return str(scoreboard[sort_col].idxmin())
