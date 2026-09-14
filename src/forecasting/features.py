"""Turn the (sector, cycle) panel into the feature table the pooled candidate trains on."""

from __future__ import annotations

import pandas as pd

SECTOR_FEATURE = "cd_setor"
CYCLE_NUMBER_FEATURE = "cycle_number"
MONTH_FEATURE = "opening_month"


def lag_columns(lags: list[int]) -> list[str]:
    return [f"lag_{lag}" for lag in lags]


def rolling_columns(windows: list[int]) -> list[str]:
    return [f"rolling_mean_{window}" for window in windows]


def feature_columns(lags: list[int], windows: list[int]) -> list[str]:
    """Every column `build_features` produces, in the order the model sees them."""
    return [
        SECTOR_FEATURE,
        CYCLE_NUMBER_FEATURE,
        MONTH_FEATURE,
        *lag_columns(lags),
        *rolling_columns(windows),
    ]


def _calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Cycle position within the year and opening month.

    Both are derived from the cycle itself, never from the sector's
    block/sub-block: block assignment is the optimizer's decision variable,
    so a forecast that used it would depend on what the optimizer is trying
    to change (see docs/agent-log).
    """
    featured = panel.copy()
    featured[CYCLE_NUMBER_FEATURE] = featured["CICLOS"].str[-2:].astype(int)
    featured[MONTH_FEATURE] = featured["opening_date"].dt.month
    return featured


def _add_lags(featured: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """One column per lag, shifted within each sector so a row never sees its own target."""
    by_sector = featured.groupby(SECTOR_FEATURE)["items"]
    for lag in lags:
        featured[f"lag_{lag}"] = by_sector.shift(lag)
    return featured


def _add_rolling_means(featured: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Rolling means over the cycles *before* each row.

    The `shift(1)` is what keeps the cycle being predicted out of its own
    features — without it, `rolling_mean_3` would average in the very value
    the model is asked to predict.
    """
    previous = featured.groupby(SECTOR_FEATURE)["items"].shift(1)
    by_sector = previous.groupby(featured[SECTOR_FEATURE])
    for window in windows:
        featured[f"rolling_mean_{window}"] = by_sector.transform(
            lambda series, w=window: series.rolling(w, min_periods=w).mean()
        )
    return featured


def build_features(panel: pd.DataFrame, lags: list[int], windows: list[int]) -> pd.DataFrame:
    """Add lag, rolling-mean and calendar features to the panel, one row per (sector, cycle).

    Rows whose lags reach back past the start of `panel` come out with NaNs
    in those columns — `build_training_table` drops them, while prediction
    rows built by a candidate keep whatever is available. The panel passed
    in must already be restricted to the training window: this function has
    no notion of a cutoff and will happily build features across whatever it
    is given.
    """
    featured = (
        _calendar_features(panel)
        .sort_values([SECTOR_FEATURE, "opening_date"])
        .reset_index(drop=True)
    )
    return _add_rolling_means(_add_lags(featured, lags), windows)


def build_training_table(
    panel: pd.DataFrame, lags: list[int], windows: list[int]
) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix and target vector for a pooled fit across every sector at once.

    Pools all sectors into one table (cross-learning): with this base's
    handful of cycles per sector, a per-sector fit has almost nothing to
    learn from, while the pooled table has thousands of rows — see the M5
    result cited in docs/papers/forecasting-volume-spec.tex section 4.5.
    """
    featured = build_features(panel, lags, windows)
    columns = feature_columns(lags, windows)
    complete = featured.dropna(subset=columns)
    return complete[columns].copy(), complete["items"].copy()
