"""Learn daily shape intensities from organization and calendar predictors."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.forecasting.dataset import CYCLE_KEYS
from src.forecasting.shape_daily import DAY_KEYS, expand_windows

ShapeFeature = Literal["CD_RE", "CD_GV", "weekday", "month"]
BASE_FEATURES = ["cd_setor", "relative_position", "n_days"]
DEFAULT_FEATURES = ["CD_RE", "CD_GV", "weekday", "month"]
CATEGORICAL = {"cd_setor", "CD_RE", "CD_GV", "weekday", "month"}


class ShapeLearningParams(BaseModel):
    """Configure a pooled LightGBM intensity model independently of level."""

    model_config = ConfigDict(extra="forbid")
    features: list[ShapeFeature] = Field(default_factory=lambda: DEFAULT_FEATURES.copy())
    n_estimators: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    num_leaves: int = Field(default=15, ge=2)
    min_child_samples: int = Field(default=30, ge=1)


def feature_table(days: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Build predictors using window metadata, organization and known calendar dates."""
    dates = pd.to_datetime(days["date"])
    frame = days.reindex(columns=["cd_setor", "n_days", "CD_RE", "CD_GV"]).copy()
    frame["relative_position"] = (days["offset"] + 0.5) / days["n_days"]
    frame["weekday"] = dates.dt.dayofweek
    frame["month"] = dates.dt.month
    return frame[[*BASE_FEATURES, *dict.fromkeys(features)]]


def _encode(train: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit categorical vocabularies on training data and mark unseen values missing."""
    train, target = train.copy(), target.copy()
    for column in train.columns.intersection(list(CATEGORICAL)):
        train[column] = train[column].astype("string").astype("category")
        values = target[column].astype("string")
        target[column] = pd.Categorical(
            values.where(values.isin(train[column].cat.categories)),
            categories=train[column].cat.categories,
        )
    return train, target


def _normalize(days: pd.DataFrame, intensities: np.ndarray) -> pd.DataFrame:
    """Clip negative intensity and conserve one unit of share per target window."""
    weights = pd.Series(np.maximum(intensities, 0), index=days.index)
    totals = weights.groupby([days[key] for key in CYCLE_KEYS]).transform("sum")
    shares = (weights / totals.replace(0, np.nan)).fillna(1 / days["n_days"])
    return days[DAY_KEYS].assign(share_pred=shares, uniform_fallback=totals.eq(0))


def predict_learned_shape(
    history: pd.DataFrame,
    windows: pd.DataFrame,
    params: ShapeLearningParams,
) -> pd.DataFrame:
    """Fit historical relative intensities and normalize future daily predictions."""
    import lightgbm as lgb

    days = expand_windows(windows)
    positive = history[history["total_actual"].gt(0)]
    if positive.empty:
        return _normalize(days, np.zeros(len(days)))
    train, target = _encode(
        feature_table(positive, params.features),
        feature_table(days, params.features),
    )
    dataset = lgb.Dataset(
        train, label=positive["actual_share"] * positive["n_days"], weight=1 / positive["n_days"]
    )
    model = lgb.train(
        {
            "objective": "regression",
            "learning_rate": params.learning_rate,
            "num_leaves": params.num_leaves,
            "min_data_in_leaf": params.min_child_samples,
            "num_threads": 1,
            "seed": 42,
            "verbosity": -1,
        },
        dataset,
        num_boost_round=params.n_estimators,
    )
    return _normalize(days, model.predict(target))


def historical_windows(history: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Carry latest known organization codes forward without reading holdout assignments."""
    windows = actual[[*CYCLE_KEYS, "window_start", "n_days"]].drop_duplicates()
    columns = history.columns.intersection(["CD_RE", "CD_GV"]).tolist()
    latest = history.sort_values("opening_date").drop_duplicates("cd_setor", keep="last")
    return windows.merge(
        latest[["cd_setor", *columns]], on="cd_setor", how="left", validate="many_to_one"
    )
