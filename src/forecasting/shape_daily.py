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


def _cycle_ages(history: pd.DataFrame) -> pd.Series:
    """Rank each sector's historical cycles by recency (0 = most recent opening)."""
    cycles = history[[*CYCLE_KEYS, "opening_date"]].drop_duplicates()
    age = cycles.groupby("cd_setor")["opening_date"].rank(method="first", ascending=False) - 1
    return age.set_axis(pd.MultiIndex.from_frame(cycles[CYCLE_KEYS]))


def _learn_bins(history: pd.DataFrame, n_bins: int, decay: float = 1.0) -> pd.DataFrame:
    """Average normalized bin masses across positive-demand historical cycles.

    `decay` discounts older cycles by `decay ** age`, where age counts cycles back
    from each sector's most recent one; `decay=1.0` reduces to the equal-weight mean.
    """
    positive = history[history["total_actual"].gt(0)].copy()
    positive["bin"] = positive["offset"] * n_bins // positive["n_days"]
    masses = (
        positive.groupby([*CYCLE_KEYS, "bin"])["actual_share"]
        .sum()
        .unstack("bin", fill_value=0)
        .reindex(columns=range(n_bins), fill_value=0)
    )
    weights = decay ** _cycle_ages(positive).reindex(masses.index)
    weighted = masses.mul(weights, axis=0).groupby("cd_setor").sum()
    totals = weights.groupby(level="cd_setor").sum()
    return weighted.div(totals, axis=0)


def _sector_weights(
    history: pd.DataFrame, days: pd.DataFrame, n_bins: int, decay: float = 1.0
) -> tuple[pd.Series, pd.Series]:
    """Project historical sector-bin masses onto target days and normalize each curve."""
    shares = _learn_bins(history, n_bins, decay)
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


def _validate_model_name(model: str) -> None:
    """Require an implemented curve model."""
    if model not in {"uniform", "sector", "recency"}:
        raise ValueError("shape model must be uniform/sector/recency")


def _validate_n_bins(n_bins: int) -> None:
    """Require a positive integer bin count."""
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("shape n_bins must be a positive integer")


def _validate_decay(model: str, decay: float) -> None:
    """Require an in-range decay when the model uses recency weighting."""
    if model == "recency" and not (0 < decay <= 1):
        raise ValueError("shape recency decay must be in (0, 1]")


def _validate_model(model: str, n_bins: int, decay: float) -> None:
    """Require an implemented curve, a positive bin count and an in-range decay."""
    _validate_model_name(model)
    _validate_n_bins(n_bins)
    _validate_decay(model, decay)


def predict_shape(
    history: pd.DataFrame,
    windows: pd.DataFrame,
    model: str = "sector",
    n_bins: int = 10,
    decay: float = 1.0,
) -> pd.DataFrame:
    """Predict daily shares from historical curves and target window metadata only.

    `model="recency"` reweights historical cycles by `decay` (the slide's `lambda`,
    1.0 = plain average, i.e. the `sector` baseline); `model="sector"` always uses
    equal weighting regardless of `decay`.
    """
    _validate_model(model, n_bins, decay)
    _validate_windows(windows)
    days = expand_windows(windows[[*CYCLE_KEYS, "window_start", "n_days"]])
    days["share_pred"] = 1 / days["n_days"]
    days["uniform_fallback"] = False
    if model in {"sector", "recency"}:
        effective_decay = decay if model == "recency" else 1.0
        days["share_pred"], days["uniform_fallback"] = _sector_weights(
            history, days, n_bins, effective_decay
        )
    return days[[*DAY_KEYS, "share_pred", "uniform_fallback"]]
