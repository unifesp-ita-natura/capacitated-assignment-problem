"""A small two-series panel the new-candidate tests share."""

from __future__ import annotations

import numpy as np
import pandas as pd

DATES = pd.date_range("2025-01-01", periods=8, freq="21D")


def panel(values_a: list[float], values_b: list[float]) -> pd.DataFrame:
    rows = [
        {"serie": key, "CICLOS": f"2025{k + 1:02d}", "items": value, "opening_date": DATES[k]}
        for key, values in (("A", values_a), ("B", values_b))
        for k, value in enumerate(values)
    ]
    return pd.DataFrame(rows).assign(window_start=lambda f: f["opening_date"], cycle_days=21)


def split(frame: pd.DataFrame, n_history: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """History is the first `n_history` cycles; targets the next one, without items."""
    cutoff = DATES[n_history]
    history = frame[frame["opening_date"] < cutoff]
    targets = frame[frame["opening_date"] == cutoff].drop(columns="items")
    return history, targets


def flat(value: float, n: int = 8) -> list[float]:
    return list(np.full(n, value))
