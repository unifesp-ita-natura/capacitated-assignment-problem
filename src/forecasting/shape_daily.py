"""Build and predict normalized demand curves per sector, cycle and day."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.dataset import CYCLE_KEYS, cycle_calendar

DAY_KEYS = [*CYCLE_KEYS, "date"]


def expand_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """Return every calendar day of each sector-cycle window, including zero-sale days."""
    days = windows.loc[windows.index.repeat(windows["n_days"])].reset_index(drop=True)
    days["offset"] = days.groupby(CYCLE_KEYS).cumcount()
    days["date"] = days["window_start"] + pd.to_timedelta(days["offset"], unit="D")
    return days


def _validate_dates(complete: pd.DataFrame) -> None:
    """Require complete calendar dates without time components."""
    dates = complete[["data_pedido", "Dt Abertura", "Dt Fechamento"]]
    if dates.isna().any().any():
        raise ValueError("shape requires nonmissing order, opening and closing dates")
    if (dates != dates.apply(lambda column: column.dt.normalize())).any().any():
        raise ValueError("shape dates must be calendar dates without time components")


def _validate_items(items: pd.Series) -> None:
    """Require finite nonnegative item counts."""
    if items.isna().any() or not np.isfinite(items).all() or items.lt(0).any():
        raise ValueError("shape requires finite, nonnegative item counts")


def _validate_orders(complete: pd.DataFrame) -> None:
    """Reject invalid demand and orders outside their declared windows."""
    _validate_dates(complete)
    _validate_items(complete["total_itens_mascarado"])
    inside = complete["data_pedido"].between(complete["Dt Abertura"], complete["Dt Fechamento"])
    if not inside.all():
        raise ValueError("shape found orders outside their declared opening/closing window")


def build_shape_days(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all CDs and fill zero days for complete sector-cycle windows."""
    calendar = cycle_calendar(raw)
    complete = raw[raw["CICLOS"].isin(calendar.loc[calendar["is_complete"], "CICLOS"])]
    _validate_orders(complete)
    windows = complete.groupby(CYCLE_KEYS, as_index=False).agg(
        window_start=("Dt Abertura", "min"), window_end=("Dt Fechamento", "max")
    )
    windows["n_days"] = (windows["window_end"] - windows["window_start"]).dt.days + 1
    actuals = (
        complete.groupby([*CYCLE_KEYS, "data_pedido"], as_index=False)
        .agg(actual=("total_itens_mascarado", "sum"))
        .rename(columns={"data_pedido": "date"})
    )
    days = expand_windows(windows).merge(actuals, on=DAY_KEYS, how="left", validate="one_to_one")
    days["actual"] = days["actual"].fillna(0)
    days["total_actual"] = days.groupby(CYCLE_KEYS)["actual"].transform("sum")
    days["actual_share"] = days["actual"] / days["total_actual"].replace(0, np.nan)
    return _attach_shape_organization(days, complete).merge(
        calendar[["CICLOS", "opening_date"]], on="CICLOS"
    )


def _attach_shape_organization(days: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Preserve optional organization codes and reject conflicting cycle assignments."""
    columns = raw.columns.intersection(["CD_RE", "CD_GV"]).tolist()
    if not columns:
        return days
    grouped = raw.groupby(CYCLE_KEYS)[columns]
    if grouped.nunique().gt(1).any().any():
        raise ValueError("shape found conflicting CD_RE/CD_GV within a sector-cycle")
    return days.merge(grouped.first().reset_index(), on=CYCLE_KEYS, validate="many_to_one")


def _learn_bins(history: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Average normalized bin masses equally across positive-demand historical cycles."""
    positive = history[history["total_actual"].gt(0)].copy()
    positive["bin"] = positive["offset"] * n_bins // positive["n_days"]
    masses = positive.groupby([*CYCLE_KEYS, "bin"])["actual_share"].sum()
    return (
        masses.unstack("bin", fill_value=0)
        .reindex(columns=range(n_bins), fill_value=0)
        .groupby("cd_setor")
        .mean()
    )


def _sector_weights(
    history: pd.DataFrame, days: pd.DataFrame, n_bins: int
) -> tuple[pd.Series, pd.Series]:
    """Project historical sector-bin masses onto target days and normalize each curve."""
    shares = _learn_bins(history, n_bins)
    bins = days["offset"] * n_bins // days["n_days"]
    masses = shares.reindex(days["cd_setor"]).to_numpy()[np.arange(len(days)), bins]
    counts = days.groupby([*CYCLE_KEYS, bins])["offset"].transform("size")
    weights = pd.Series(masses, index=days.index).fillna(0) / counts
    totals = weights.groupby([days[key] for key in CYCLE_KEYS]).transform("sum")
    return (weights / totals.replace(0, np.nan)).fillna(1 / days["n_days"]), totals.eq(0)


def _validate_windows(windows: pd.DataFrame) -> None:
    """Require unique windows with positive integer day counts and valid starts."""
    if windows[CYCLE_KEYS].isna().any().any() or windows[CYCLE_KEYS].duplicated().any():
        raise ValueError("shape requires unique nonmissing sector-cycle keys")
    lengths = windows["n_days"]
    valid_lengths = np.isfinite(lengths) & lengths.ge(1) & lengths.eq(lengths.round())
    if not valid_lengths.all() or windows["window_start"].isna().any():
        raise ValueError("shape requires valid window starts and positive integer day counts")


def _validate_model(model: str, n_bins: int) -> None:
    """Require an implemented curve and a positive integer number of bins."""
    if model not in {"uniform", "sector"}:
        raise ValueError("shape model must be uniform/sector")
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("shape n_bins must be a positive integer")


def predict_shape(
    history: pd.DataFrame, windows: pd.DataFrame, model: str = "sector", n_bins: int = 10
) -> pd.DataFrame:
    """Predict daily shares from historical curves and target window metadata only."""
    _validate_model(model, n_bins)
    _validate_windows(windows)
    days = expand_windows(windows[[*CYCLE_KEYS, "window_start", "n_days"]])
    days["share_pred"] = 1 / days["n_days"]
    days["uniform_fallback"] = False
    if model == "sector":
        days["share_pred"], days["uniform_fallback"] = _sector_weights(history, days, n_bins)
    return days[[*DAY_KEYS, "share_pred", "uniform_fallback"]]
