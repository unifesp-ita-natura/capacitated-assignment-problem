"""Shared error-metric and scoreboard helpers for ranking forecast strategies."""

from __future__ import annotations

import pandas as pd


def mae_wmape(comparison: pd.DataFrame, forecast_col: str, actual_col: str):
    error = comparison[forecast_col] - comparison[actual_col]
    mae = float(error.abs().mean())
    wmape = float(error.abs().sum() / comparison[actual_col].sum())
    return mae, wmape


def build_scoreboard(scores: list[dict], sort_col: str) -> pd.DataFrame:
    return pd.DataFrame(scores).set_index("strategy").sort_values(sort_col)


def select_best(scoreboard: pd.DataFrame, sort_col: str) -> str:
    return str(scoreboard[sort_col].idxmin())
