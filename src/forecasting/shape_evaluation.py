"""Evaluate daily shape independently of level using rolling historical origins."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.forecasting.dataset import CYCLE_KEYS
from src.forecasting.shape_daily import DAY_KEYS, predict_shape
from src.forecasting.shape_learning import (
    ShapeLearningParams,
    historical_windows,
    predict_learned_shape,
)


class ShapeCandidate(BaseModel):
    """Parameters for one independently named daily curve candidate."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    model: Literal["uniform", "sector", "lightgbm"]
    n_bins: int = Field(default=10, ge=1)
    learning: ShapeLearningParams = Field(default_factory=ShapeLearningParams)


class ShapeSplit(BaseModel):
    """Cycle-based rolling-origin parameters for shape evaluation."""

    model_config = ConfigDict(extra="forbid")
    min_train_cycles: int = Field(default=20, ge=1)
    horizon: int = Field(default=1, ge=1)
    window: Literal["expanding", "sliding"] = "expanding"


def _score_origin(
    days: pd.DataFrame,
    train: list[str],
    targets: list[str],
    candidate: ShapeCandidate,
) -> pd.DataFrame:
    """Use historical shares for prediction and actual target totals only for evaluation."""
    history = days[days["CICLOS"].isin(train)]
    actual = days[days["CICLOS"].isin(targets)]
    windows = actual[[*CYCLE_KEYS, "window_start", "n_days"]].drop_duplicates()
    forecast = _predict_candidate(history, actual, windows, candidate)
    scored = actual.merge(forecast, on=DAY_KEYS, validate="one_to_one")
    scored["items_pred"] = scored["share_pred"] * scored["total_actual"]
    scored["error"] = scored["items_pred"] - scored["actual"]
    scored["abs_error"] = scored["error"].abs()
    scored["abs_share_error"] = (scored["share_pred"] - scored["actual_share"]).abs()
    return scored.assign(candidate=candidate.name, origin_cycle=train[-1])


def _predict_candidate(history, actual, windows, candidate):
    """Dispatch shape strategies while keeping holdout demand out of learned predictors."""
    if candidate.model == "lightgbm":
        return predict_learned_shape(
            history, historical_windows(history, actual), candidate.learning
        )
    return predict_shape(history, windows, candidate.model, candidate.n_bins)


def _validate_candidates(candidates: list[ShapeCandidate]) -> None:
    """Require at least one candidate with unique names."""
    if not candidates or len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("shape candidates must have distinct names and cannot be empty")


def _ordered_cycles(days: pd.DataFrame) -> list[str]:
    """Order complete observed cycles by their shared opening date."""
    return (
        days[["CICLOS", "opening_date"]]
        .drop_duplicates()
        .sort_values(["opening_date", "CICLOS"])["CICLOS"]
        .tolist()
    )


def _origins(cycles: list[str], split: ShapeSplit) -> range:
    """Return origins with enough historical and held-out cycles."""
    origins = range(split.min_train_cycles, len(cycles) - split.horizon + 1)
    if not origins:
        raise ValueError("not enough complete cycles for the configured shape backtest")
    return origins


def _split_at(cycles: list[str], index: int, split: ShapeSplit) -> tuple[list[str], list[str]]:
    """Select expanding or sliding history and the following target cycles."""
    start = 0 if split.window == "expanding" else index - split.min_train_cycles
    return cycles[start:index], cycles[index : index + split.horizon]


def evaluate_shapes(
    days: pd.DataFrame,
    candidates: list[ShapeCandidate],
    split: ShapeSplit,
) -> pd.DataFrame:
    """Evaluate every candidate on identical held-out cycle windows."""
    _validate_candidates(candidates)
    cycles = _ordered_cycles(days)
    results = []
    for index in _origins(cycles, split):
        train, targets = _split_at(cycles, index, split)
        results.extend(_score_origin(days, train, targets, candidate) for candidate in candidates)
    return pd.concat(results, ignore_index=True)


def _candidate_metrics(scored: pd.DataFrame) -> dict:
    """Report daily volume diagnostics and equally weighted sector-cycle shape errors."""
    positive = scored[scored["total_actual"].gt(0)]
    curve_keys = ["origin_cycle", *CYCLE_KEYS]
    curves = positive.groupby(curve_keys)["abs_share_error"]
    total = scored["actual"].sum()
    return {
        "share_mae_pp": 100 * curves.mean().mean(),
        "total_variation": 0.5 * curves.sum().mean(),
        "mae": scored["abs_error"].mean(),
        "rmse": np.sqrt(scored["error"].pow(2).mean()),
        "bias": scored["error"].mean(),
        "bias_pct": 100 * scored["error"].sum() / total if total else np.nan,
        "wmape": scored["abs_error"].sum() / total if total else np.nan,
        "n_days": len(scored),
        "n_sector_cycles": len(scored[curve_keys].drop_duplicates()),
        "n_zero_total_cycles": len(
            scored.loc[scored["total_actual"].eq(0), curve_keys].drop_duplicates()
        ),
        "n_uniform_fallback_cycles": len(
            scored.loc[scored["uniform_fallback"], curve_keys].drop_duplicates()
        ),
        "n_origins": scored["origin_cycle"].nunique(),
    }


def compare_shapes(predictions: pd.DataFrame) -> pd.DataFrame:
    """Rank shape candidates by share MAE in percentage points."""
    rows = [
        dict(candidate=name, **_candidate_metrics(group))
        for name, group in predictions.groupby("candidate", sort=False)
    ]
    return pd.DataFrame(rows).sort_values("share_mae_pp", na_position="last").reset_index(drop=True)
