"""Combine predicted city-cycle totals with sector daily shapes for explicit opening scenarios."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

import src.forecasting.candidates  # noqa: F401
from src.config.schema import CombinedLevelParams
from src.forecasting.combined_inputs import (
    WINDOW_KEYS,
    Histories,
    prepare_histories,
    prepare_windows,
    validate_known_gaps,
    validate_path,
)
from src.forecasting.hierarchy_config import HierarchyForecastParams
from src.forecasting.hierarchy_future import forecast_hierarchical_path
from src.forecasting.model import REGISTRY, forecast
from src.forecasting.shape_daily import predict_shape
from src.forecasting.shape_evaluation import ShapeCandidate
from src.forecasting.shape_learning import historical_windows, predict_learned_shape

LEVEL_KEYS = ["scenario_id", "serie", "CICLOS"]


class CombinedResult(NamedTuple):
    """Intermediate forecasts and the final city-day demand table."""

    level: pd.DataFrame
    shape: pd.DataFrame
    daily: pd.DataFrame
    training_cycles: list[str]


def _level_queries(history: Histories, windows: pd.DataFrame) -> pd.DataFrame:
    """Expand sector scenarios to cities known before the cutoff."""
    unknown = set(windows["cd_setor"]) - set(history.cities["cd_setor"])
    if unknown:
        raise ValueError(f"no known city history for sectors: {sorted(unknown)}")
    return windows.merge(history.cities, on="cd_setor", validate="many_to_many")


def _shape_forecast(
    history: Histories, windows: pd.DataFrame, params: ShapeCandidate
) -> pd.DataFrame:
    """Forecast one daily curve per sector-cycle without mixing scenario paths."""
    if params.model == "lightgbm":
        _require_shape_features(history.shape, params)
        shares = predict_learned_shape(
            history.shape, historical_windows(history.shape, windows), params.learning
        )
    else:
        shares = predict_shape(history.shape, windows, params.model, params.n_bins)
    columns = [*WINDOW_KEYS, "opening_date", "window_start", "window_end", "n_days", "emit"]
    result = shares.merge(windows[columns], on=["cd_setor", "CICLOS"], validate="many_to_one")
    result["relative_day"] = (result["date"] - result["window_start"]).dt.days + 1
    return result


def _require_shape_features(history: pd.DataFrame, params: ShapeCandidate) -> None:
    """Reject missing organization inputs rather than silently disabling requested predictors."""
    missing = set(params.learning.features).intersection({"CD_RE", "CD_GV"}) - set(history.columns)
    if missing:
        raise ValueError(f"missing shape features in history: {sorted(missing)}")


def _validate_shares(shape: pd.DataFrame) -> None:
    """Require unique daily shares that are finite, nonnegative and sum to one."""
    if shape[[*WINDOW_KEYS, "date"]].duplicated().any():
        raise ValueError("duplicate daily shape keys")
    valid = np.isfinite(shape["share_pred"]) & shape["share_pred"].ge(0)
    totals = shape.groupby(WINDOW_KEYS)["share_pred"].sum()
    if not valid.all() or not np.allclose(totals, 1):
        raise ValueError("shape shares must be finite, nonnegative and sum to one")


def combine_predictions(level: pd.DataFrame, shape: pd.DataFrame) -> pd.DataFrame:
    """Multiply city totals by aligned daily shares without using future actual demand."""
    _validate_shares(shape)
    _validate_levels(level)
    keys = [*WINDOW_KEYS, "window_start", "window_end"]
    daily = level.merge(
        shape[[*keys, "date", "relative_day", "share_pred", "uniform_fallback"]],
        on=keys,
        how="left",
        validate="many_to_many",
    )
    if daily["share_pred"].isna().any():
        raise ValueError("a level forecast has no matching shape window")
    daily["items_pred"] = daily["level_pred"] * daily["share_pred"]
    _validate_conservation(level, daily)
    return daily


def _validate_levels(level: pd.DataFrame) -> None:
    """Preserve missing model forecasts explicitly while rejecting invalid city totals."""
    if level[LEVEL_KEYS].duplicated().any():
        raise ValueError("duplicate level keys")
    known = level["level_pred"].dropna()
    if not (np.isfinite(known) & known.ge(0)).all():
        raise ValueError("level forecasts must be finite and nonnegative, or missing")


def _validate_conservation(level: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Ensure daily amounts preserve each predicted city-cycle total."""
    totals = daily.groupby(LEVEL_KEYS)["items_pred"].sum(min_count=1)
    expected = level.set_index(LEVEL_KEYS)["level_pred"].reindex(totals.index)
    if not np.allclose(totals, expected, equal_nan=True):
        raise ValueError("daily predictions do not conserve city-cycle totals")


def _forecast_path(
    history: Histories,
    windows: pd.DataFrame,
    level_params: CombinedLevelParams,
    shape_params: ShapeCandidate,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep recursive level forecasts and daily curve normalization within one scenario path."""
    queries = _level_queries(history, windows)
    level = _forecast_level(history, queries, level_params)
    level["level_status"] = np.where(level["level_pred"].isna(), "missing", "ok")
    return level, _shape_forecast(history, windows, shape_params)


def _forecast_level(history, queries, params):
    if isinstance(params, HierarchyForecastParams):
        return forecast_hierarchical_path(history.hierarchy, queries, params, len(history.cycles))
    return forecast(REGISTRY.build(params), history.level, queries).rename(
        columns={"items_pred": "level_pred"}
    )


def forecast_combined(
    raw: pd.DataFrame,
    windows: pd.DataFrame,
    as_of,
    history_cycles: int,
    level_params: CombinedLevelParams,
    shape_params: ShapeCandidate,
) -> CombinedResult:
    """Run point-in-time city-level and sector-shape models and return emitted daily forecasts."""
    timestamp = pd.Timestamp(as_of)
    _validate_timestamp(timestamp, history_cycles)
    windows = prepare_windows(windows)
    history = prepare_histories(raw, timestamp, history_cycles)
    results = []
    for _, path in windows.groupby("scenario_id", sort=False):
        validate_path(path, timestamp)
        _require_new_cycles(history, path)
        validate_known_gaps(raw, timestamp, history, path)
        results.append(_forecast_path(history, path, level_params, shape_params))
    level = pd.concat([result[0] for result in results], ignore_index=True)
    shape = pd.concat([result[1] for result in results], ignore_index=True)
    level, shape = level[level["emit"]].copy(), shape[shape["emit"]].copy()
    return CombinedResult(level, shape, combine_predictions(level, shape), history.cycles)


def _require_new_cycles(history: Histories, windows: pd.DataFrame) -> None:
    """Prevent reusing a training cycle code with a different future date."""
    if set(history.level["CICLOS"]).intersection(windows["CICLOS"]):
        raise ValueError("scenario cycle codes must be after the training cycles")


def _validate_timestamp(timestamp: pd.Timestamp, history_cycles: int) -> None:
    """Use explicit local timestamps and a positive number of historical cycles."""
    if pd.isna(timestamp) or timestamp.tzinfo is not None:
        raise ValueError("as_of must be a valid local timestamp without timezone suffix")
    if history_cycles < 1:
        raise ValueError("history_cycles must be positive")
