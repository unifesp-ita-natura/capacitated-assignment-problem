"""Forecast the held-out horizon cycles from raw data with a point-in-time level x shape model."""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from src.config.schema import CombinedLevelParams, PipelineConfig
from src.forecasting.combined import CombinedResult, forecast_combined
from src.forecasting.combined_inputs import prepare_histories
from src.forecasting.combined_validation import recorded_windows, validate_combined
from src.forecasting.dataset import cycle_calendar
from src.forecasting.shape_evaluation import ShapeCandidate


class HorizonPlan(NamedTuple):
    """Forecast cutoff, the cycles to forecast, and the sector windows of the whole path."""

    as_of: pd.Timestamp
    targets: list[str]
    windows: pd.DataFrame


def shape_candidate(config: PipelineConfig) -> ShapeCandidate:
    """The daily-shape model the pipeline config asks for."""
    name = f"{config.shape_model}_{config.shape_bins}_bins"
    return ShapeCandidate(name=name, model=config.shape_model, n_bins=config.shape_bins)


def pick_targets(
    calendar: pd.DataFrame, horizon_cycles: int, history_cycles: int, shift: int = 0
) -> list[str]:
    """The last `horizon_cycles` complete cycles, leaving `history_cycles` of them to train on.

    `shift` moves the horizon back by that many complete cycles (0 = the most recent window).
    """
    complete = calendar[calendar["is_complete"]].sort_values("opening_date")
    needed = history_cycles + horizon_cycles + shift
    if len(complete) < needed:
        raise ValueError(
            f"need {needed} complete cycles ({history_cycles} history + {horizon_cycles} "
            f"horizon + {shift} shift); found {len(complete)}"
        )
    kept = complete["CICLOS"].iloc[: len(complete) - shift]
    return kept.tail(horizon_cycles).tolist()


def _complete_sectors(windows: pd.DataFrame, known: set[str]) -> pd.DataFrame:
    """Keep sectors with a window in every path cycle and a known city history."""
    n_cycles = windows["CICLOS"].nunique()
    per_sector = windows.groupby("cd_setor")["CICLOS"].nunique()
    keep = set(per_sector[per_sector.eq(n_cycles)].index) & known
    return windows[windows["cd_setor"].isin(keep)].copy()


def plan_horizon(
    raw: pd.DataFrame, horizon_cycles: int, history_cycles: int, shift: int = 0
) -> HorizonPlan:
    """Cut off just before the first target cycle opens and list every cycle windows to emit."""
    calendar = cycle_calendar(raw)
    targets = pick_targets(calendar, horizon_cycles, history_cycles, shift)
    opening = calendar.set_index("CICLOS").loc[targets, "opening_date"]
    as_of = opening.min() - pd.Timedelta(days=1)
    history = prepare_histories(raw, as_of, history_cycles)
    path = calendar[
        calendar["opening_date"].gt(history.level["opening_date"].max())
        & calendar["opening_date"].le(opening.max())
    ]
    windows = recorded_windows(raw, path["CICLOS"].tolist(), targets, "horizon")
    return HorizonPlan(as_of, targets, _complete_sectors(windows, set(history.cities["cd_setor"])))


def forecast_horizon(
    raw: pd.DataFrame, plan: HorizonPlan, config: PipelineConfig, level: CombinedLevelParams
) -> CombinedResult:
    """Daily item forecasts for the plan's target cycles, using only data before `as_of`."""
    return forecast_combined(
        raw, plan.windows, plan.as_of, config.history_cycles, level, shape_candidate(config)
    )


def forecast_accuracy(
    raw: pd.DataFrame, plan: HorizonPlan, config: PipelineConfig, level: CombinedLevelParams
) -> dict[str, float]:
    """Overall WMAPE, bias and volume coverage of the same models replayed on the target cycles."""
    result = validate_combined(
        raw, plan.targets, config.history_cycles, level, shape_candidate(config)
    )
    overall = result.metrics.set_index("scope").loc["overall"]
    return {
        name: float(overall[name])
        for name in ["wmape", "bias_pct", "mae", "actual_volume_coverage"]
    }
