"""Fit shared temporal totals, historical shares and pooled internal LightGBM models."""

import numpy as np
import pandas as pd

from src.forecasting.candidates.ets import _smooth
from src.forecasting.hierarchy_config import HierarchySettings, LearningSettings
from src.forecasting.hierarchy_features import feature_columns


def moving_average(history: pd.DataFrame, query: pd.DataFrame, window: int) -> np.ndarray:
    """Average at most K available node observations, retaining valid zeros."""
    recent = history.sort_values("opening_date").groupby("node").tail(window)
    means = recent.groupby("node")["items"].mean()
    return query["node"].map(means).fillna(0).to_numpy(float)


def ets_total(
    history: pd.DataFrame, query: pd.DataFrame, config: HierarchySettings
) -> tuple[np.ndarray, dict]:
    """Select shared alpha and duration exponent using training-only one-step absolute errors."""
    matrix = history.pivot(index="node", columns="opening_date", values="items").sort_index(axis=1)
    days = history.pivot(index="node", columns="opening_date", values="cycle_days").reindex_like(
        matrix
    )
    values, durations = matrix.to_numpy(float), days.to_numpy(float)
    settings = [(alpha, beta) for alpha in config.ets_alphas for beta in [0.0, 0.5, 1.0]]
    errors = [_smooth(values, alpha, durations**beta)[1] for alpha, beta in settings]
    alpha, beta = settings[int(np.argmin(errors))]
    levels = pd.Series(_smooth(values, alpha, durations**beta)[0], index=matrix.index)
    predicted = (
        query["node"].map(levels).fillna(0).to_numpy() * query["cycle_days"].to_numpy() ** beta
    )
    return np.maximum(predicted, 0), {"alpha": alpha, "duration_exponent": beta}


def normalize_scores(query: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Normalize nonnegative scores among siblings with an explicit uniform zero-score fallback."""
    values = pd.Series(np.maximum(scores, 0), index=query.index).fillna(0)
    totals = values.groupby(query["parent"]).transform("sum")
    count = query.groupby("parent")["node"].transform("size")
    return (values / totals.replace(0, np.nan)).fillna(1 / count).to_numpy()


def historical_shares(
    training: pd.DataFrame, query: pd.DataFrame, window: int, prior: float
) -> np.ndarray:
    """Shrink recent ratios of sums toward the full available historical ratio."""
    recent = training.sort_values("opening_date").groupby("node").tail(window)
    sums = recent.groupby("node")[["items", "parent_actual"]].sum()
    full = training.groupby("node")[["items", "parent_actual"]].sum()
    ratio = sums["items"] / sums["parent_actual"].replace(0, np.nan)
    long_ratio = full["items"] / full["parent_actual"].replace(0, np.nan)
    n = recent.groupby("node")["items"].count()
    weight = n / (n + prior)
    blended = (ratio * weight + long_ratio * (1 - weight)).fillna(long_ratio)
    return normalize_scores(query, query["node"].map(blended).fillna(0).to_numpy())


def _encoded_frames(
    training: pd.DataFrame, query: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, target = training[columns].copy(), query[columns].copy()
    for column in ["node", "parent"]:
        dtype = pd.CategoricalDtype(sorted(train[column].unique()))
        train[column] = train[column].astype(dtype)
        target[column] = target[column].astype(dtype)
    return train, target


def _training_rows(training: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, pd.Series]:
    usable = training[training["n_history"].ge(1)].copy()
    labels = usable["items"]
    if kind == "share":
        usable = usable[usable["parent_actual"].gt(0)]
        labels = usable["items"] / usable["parent_actual"]
    return usable, labels


def _settings(config: LearningSettings, kind: str) -> dict:
    objective = "regression" if kind == "share" else config.volume_objective
    return {
        "objective": objective,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_child_samples,
        "seed": config.random_state,
        "num_threads": 1,
        "verbosity": -1,
    }


def learned_values(
    training: pd.DataFrame,
    query: pd.DataFrame,
    config: LearningSettings,
    kind: str,
    fallback: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit one model for all nodes in this level; weak city histories borrow from the fallback."""
    import lightgbm as lgb

    usable, labels = _training_rows(training, kind)
    columns = feature_columns(config.features)
    if usable.empty or labels.sum() == 0:
        return fallback.copy(), {"model": "historical_fallback", "n_train": len(usable)}
    train, target = _encoded_frames(usable, query, columns)
    model = lgb.train(
        _settings(config, kind), lgb.Dataset(train, label=labels), config.n_estimators
    )
    values = np.maximum(model.predict(target, num_threads=1), 0)
    if query["level"].iloc[0] == "city":
        counts = training[training["items"].gt(0)].groupby("node").size()
        n = query["node"].map(counts).fillna(0).to_numpy()
        weight = n / (n + config.city_prior_cycles)
        values = weight * values + (1 - weight) * fallback
    fitted = {
        "model": "lightgbm",
        "target": kind,
        "objective": _settings(config, kind)["objective"],
        "n_train": len(usable),
        "features": columns,
        "feature_importance": dict(zip(columns, model.feature_importance().tolist())),
    }
    return values, fitted
