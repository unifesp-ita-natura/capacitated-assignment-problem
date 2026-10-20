"""Spread per-cycle item forecasts over each window's days; score them per sector-day and CD-day."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

from src.forecasting.dataset import DailyBase

CYCLE_KEYS = ["cd_setor", "CICLOS"]
DAY_KEYS = ["cd_setor", "cd_cd", "CICLOS", "date"]
N_BINS = 10  # a window is cut into tenths, so 14- and 23-day windows share one curve

Curve = Literal["uniform", "sector"]


def _bin_of(offset: pd.Series, n_days: pd.Series) -> pd.Series:
    """Which tenth of its window a day falls in (0..N_BINS-1)."""
    return offset * N_BINS // n_days


def sector_bin_shares(actuals: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Each sector's average share of a cycle's items falling in each tenth of the window."""
    days = actuals.merge(windows, on=CYCLE_KEYS)
    days["bin"] = _bin_of((days["date"] - days["window_start"]).dt.days, days["n_days"])
    per_cycle = days.groupby([*CYCLE_KEYS, "bin"])["items"].sum()
    per_cycle = per_cycle / per_cycle.groupby(CYCLE_KEYS).transform("sum")
    shares = per_cycle.unstack("bin", fill_value=0).groupby("cd_setor").mean()
    return shares.reindex(columns=range(N_BINS), fill_value=0)


def sector_cd_shares(actuals: pd.DataFrame) -> pd.DataFrame:
    """Each sector's share of its items shipped from each CD, over the given history."""
    by_cd = actuals.groupby(["cd_setor", "cd_cd"], as_index=False)["items"].sum()
    by_cd["cd_share"] = by_cd["items"] / by_cd.groupby("cd_setor")["items"].transform("sum")
    return by_cd[["cd_setor", "cd_cd", "cd_share"]]


def _uniform_weight(days: pd.DataFrame, shares: pd.DataFrame) -> pd.Series:
    return 1 / days["n_days"]


def _sector_weight(days: pd.DataFrame, shares: pd.DataFrame) -> pd.Series:
    """The sector's own curve; a tenth's share is split evenly over its days in this window."""
    bins = _bin_of(days["offset"], days["n_days"])
    share = shares.reindex(days["cd_setor"]).to_numpy()[np.arange(len(days)), bins]
    weight = pd.Series(share, index=days.index)
    weight = weight / bins.groupby([days["cd_setor"], days["CICLOS"], bins]).transform("size")
    return weight / weight.groupby([days["cd_setor"], days["CICLOS"]]).transform("sum")


CURVES: dict[str, Callable[[pd.DataFrame, pd.DataFrame], pd.Series]] = {
    "uniform": _uniform_weight,
    "sector": _sector_weight,
}


def _one_row_per_day(predictions: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    days = predictions.merge(windows, on=CYCLE_KEYS)
    days = days.loc[days.index.repeat(days["n_days"])].reset_index(drop=True)
    days["offset"] = days.groupby(CYCLE_KEYS).cumcount()
    days["date"] = days["window_start"] + pd.to_timedelta(days["offset"], unit="D")
    return days


def spread(
    predictions: pd.DataFrame,
    windows: pd.DataFrame,
    shares: pd.DataFrame,
    cd_shares: pd.DataFrame,
    curve: Curve,
) -> pd.DataFrame:
    """Per-(sector, CD, cycle, date) forecast: cycle total × the day's weight × the CD's share."""
    days = _one_row_per_day(predictions, windows)
    days["pred"] = days["items_pred"] * CURVES[curve](days, shares)
    days = days.merge(cd_shares, on="cd_setor")
    days["pred"] *= days["cd_share"]
    return days[[*DAY_KEYS, "pred"]]


def score_daily(
    predictions: pd.DataFrame, daily: DailyBase, train_cycles: list[str], curve: Curve
) -> pd.DataFrame:
    """Predicted vs actual items per (sector, CD, cycle, date) for the predicted (sector, cycle)s.

    The curve and the CD split are learned from `train_cycles` alone. Days
    with no order count as an actual of zero, and a (CD, day) only the
    forecast reaches counts as a prediction against zero.
    """
    history = daily.actuals[daily.actuals["CICLOS"].isin(train_cycles)]
    shares = sector_bin_shares(history, daily.windows)
    forecast = spread(predictions, daily.windows, shares, sector_cd_shares(history), curve)

    actual = daily.actuals.merge(predictions[CYCLE_KEYS], on=CYCLE_KEYS)
    both = forecast.merge(actual.rename(columns={"items": "actual"}), on=DAY_KEYS, how="outer")
    return both.fillna({"pred": 0.0, "actual": 0.0})


def _abs_error_by(daily: pd.DataFrame, keys: list[str]) -> pd.Series:
    totals = daily.groupby(keys)[["actual", "pred"]].sum()
    return (totals["actual"] - totals["pred"]).abs()


def sector_day_mae(daily: pd.DataFrame) -> float:
    """Mean |actual - pred| per (sector, day), each sector counting once."""
    return float(_abs_error_by(daily, ["cd_setor", "date"]).groupby("cd_setor").mean().mean())


def cd_day_mae(daily: pd.DataFrame) -> float:
    """Mean |actual - pred| per (CD, day), each CD counting once — where capacity acts."""
    return float(_abs_error_by(daily, ["cd_cd", "date"]).groupby("cd_cd").mean().mean())


def cd_day_wape(daily: pd.DataFrame) -> float:
    """Total CD-day error as a share of total CD-day items — the scale-free companion of the MAE."""
    error = _abs_error_by(daily, ["cd_cd", "date"])
    return float(error.sum() / daily["actual"].sum())
