"""Turn the (sector, cycle) panel into the feature table the pooled candidate trains on."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

SECTOR_FEATURE = "cd_setor"
CYCLE_NUMBER_FEATURE = "cycle_number"
MONTH_FEATURE = "opening_month"

# Series the panel carries alongside `items`, lagged as features of their
# own when `companion_lags` asks for them. An order count is roughly "how
# many consultants bought at all", which moves far less from cycle to cycle
# than the item count does — measured at Spearman 0.230 against next
# cycle's items, against 0.180 for the item count itself. See
# docs/plans/forecasting-feature-roadmap.md, stage 1.
COMPANION_SERIES = ("orders", "volumes")
ITEMS_PER_ORDER_FEATURE = "items_per_order"

# Each row's running mean of its sector's EARLIER cycles — exactly what the
# naive:mean candidate forecasts for that row. Never a feature: it is the
# denominator a ratio target divides by, so a model trained on the ratio is
# predicting a correction on top of that benchmark.
LEVEL_REFERENCE = "level_ref"


def lag_columns(lags: Sequence[int]) -> list[str]:
    return [f"lag_{lag}" for lag in lags]


def rolling_columns(windows: Sequence[int]) -> list[str]:
    return [f"rolling_mean_{window}" for window in windows]


def companion_columns(lags: Sequence[int]) -> list[str]:
    """Lagged order and volume counts, plus the items-per-order ratio a tree can't derive itself.

    A gradient-boosted tree splits on one column at a time, so it can
    represent "orders above 40" but never "items divided by orders". The
    ratio is the part of the item count that the order count doesn't
    already explain — cart size — so it's handed over precomputed.
    """
    lagged = [f"{series}_lag_{lag}" for series in COMPANION_SERIES for lag in lags]
    return lagged + [f"{ITEMS_PER_ORDER_FEATURE}_lag_{lag}" for lag in lags]


def feature_columns(
    lags: Sequence[int], windows: Sequence[int], companion_lags: Sequence[int] = ()
) -> list[str]:
    """Every column `build_features` produces, in the order the model sees them."""
    return [
        SECTOR_FEATURE,
        CYCLE_NUMBER_FEATURE,
        MONTH_FEATURE,
        *lag_columns(lags),
        *rolling_columns(windows),
        *companion_columns(companion_lags),
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


def _add_lags(featured: pd.DataFrame, lags: Sequence[int]) -> pd.DataFrame:
    """One column per lag, shifted within each sector so a row never sees its own target."""
    by_sector = featured.groupby(SECTOR_FEATURE)["items"]
    for lag in lags:
        featured[f"lag_{lag}"] = by_sector.shift(lag)
    return featured


def _add_rolling_means(featured: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
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


def _add_companion_lags(featured: pd.DataFrame, lags: Sequence[int]) -> pd.DataFrame:
    """Lag the order and volume counts within each sector, same shift rule as the item lags."""
    for series in COMPANION_SERIES:
        by_sector = featured.groupby(SECTOR_FEATURE)[series]
        for lag in lags:
            featured[f"{series}_lag_{lag}"] = by_sector.shift(lag)
    return _add_items_per_order_lags(featured, lags)


def _add_level_reference(featured: pd.DataFrame) -> pd.DataFrame:
    """Running mean of each sector's earlier cycles, shifted so a row never sees its own value."""
    previous = featured.groupby(SECTOR_FEATURE)["items"].shift(1)
    featured[LEVEL_REFERENCE] = previous.groupby(featured[SECTOR_FEATURE]).transform(
        lambda series: series.expanding().mean()
    )
    return featured


def _add_items_per_order_lags(featured: pd.DataFrame, lags: Sequence[int]) -> pd.DataFrame:
    """Items per order in each earlier cycle, computed before shifting so no row sees its own."""
    ratio = (featured["items"] / featured["orders"]).replace([np.inf, -np.inf], np.nan)
    by_sector = ratio.groupby(featured[SECTOR_FEATURE])
    for lag in lags:
        featured[f"{ITEMS_PER_ORDER_FEATURE}_lag_{lag}"] = by_sector.shift(lag)
    return featured


def build_features(
    panel: pd.DataFrame,
    lags: Sequence[int],
    windows: Sequence[int],
    companion_lags: Sequence[int] = (),
) -> pd.DataFrame:
    """Add lag, rolling-mean and calendar features to the panel, one row per (sector, cycle).

    Rows whose lags reach back past the start of `panel` come out with NaNs
    in those columns — `build_training_table` drops them, while prediction
    rows built by a candidate keep whatever is available. The panel passed
    in must already be restricted to the training window: this function has
    no notion of a cutoff and will happily build features across whatever it
    is given. `companion_lags` is empty by default, which keeps the table
    exactly as it was before order counts reached the panel.
    """
    featured = (
        _calendar_features(panel)
        .sort_values([SECTOR_FEATURE, "opening_date"])
        .reset_index(drop=True)
    )
    featured = _add_level_reference(_add_rolling_means(_add_lags(featured, lags), windows))
    if not companion_lags:
        return featured
    return _add_companion_lags(featured, companion_lags)


def complete_feature_frame(
    panel: pd.DataFrame,
    lags: Sequence[int],
    windows: Sequence[int],
    companion_lags: Sequence[int] = (),
) -> pd.DataFrame:
    """Rows a pooled model can train on: featured, with every model column present.

    Pools all sectors into one table (cross-learning): with this base's
    handful of cycles per sector, a per-sector fit has almost nothing to
    learn from, while the pooled table has thousands of rows — see the M5
    result cited in docs/papers/forecasting-volume-spec.tex section 4.5.
    Rows missing any feature are dropped, so asking for companion lags
    deeper than the item lags costs training rows. The frame keeps its
    non-feature columns (`items`, `CICLOS`, `opening_date`, `level_ref`), so
    a caller can cut a target, a validation split or a ratio denominator
    from the same rows the features came from.
    """
    featured = build_features(panel, lags, windows, companion_lags)
    return featured.dropna(subset=feature_columns(lags, windows, companion_lags))
