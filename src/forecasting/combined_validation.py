"""Replay a fixed level-shape configuration and measure held-out city-day demand errors."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.config.schema import ForecastParams
from src.forecasting.combined import forecast_combined
from src.forecasting.combined_inputs import prepare_histories
from src.forecasting.dataset import cycle_calendar
from src.forecasting.metrics import bias, bias_pct, mae, rmse, wmape
from src.forecasting.shape_evaluation import ShapeCandidate

ACTUAL_KEYS = ["scenario_id", "cd_setor", "estado", "cidade", "CICLOS", "date"]


class ValidationResult(NamedTuple):
    """Daily comparisons, fixed-model diagnostics and the historical scenario schedule."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    windows: pd.DataFrame


def recorded_windows(
    raw: pd.DataFrame, cycles: list[str], targets: list[str], scenario_id: str
) -> pd.DataFrame:
    """Reconstruct recorded sector windows without supplying any holdout counts to models."""
    selected = raw[raw["CICLOS"].isin(cycles)]
    windows = selected.groupby(["cd_setor", "CICLOS"], as_index=False).agg(
        window_start=("Dt Abertura", "min"), window_end=("Dt Fechamento", "max")
    )
    windows = windows.merge(cycle_calendar(raw)[["CICLOS", "opening_date"]], on="CICLOS")
    windows["scenario_id"] = scenario_id
    windows["emit"] = windows["CICLOS"].isin(targets)
    return windows


def _replay_windows(raw, calendar, target, history):
    """Keep forecast paths with known cities and recorded windows for all intermediate steps."""
    opening = calendar.loc[calendar["CICLOS"].eq(target), "opening_date"].iloc[0]
    future = calendar[
        calendar["opening_date"].gt(history.level["opening_date"].max())
        & calendar["opening_date"].le(opening)
    ]
    windows = recorded_windows(raw, future["CICLOS"].tolist(), [target], f"observed_{target}")
    complete = windows.groupby("cd_setor")["CICLOS"].nunique().eq(len(future))
    known = set(history.cities["cd_setor"]).intersection(complete[complete].index)
    return windows[windows["cd_setor"].isin(known)].copy(), windows[windows["emit"]].copy()


def _actual_days(raw: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate actual demand at city-day resolution for the observed target scenario."""
    actual = (
        raw.groupby(["cd_setor", "estado", "cidade", "CICLOS", "data_pedido"], as_index=False)
        .agg(actual=("total_itens_mascarado", "sum"))
        .rename(columns={"data_pedido": "date"})
    )
    return actual.merge(
        windows[["scenario_id", "cd_setor", "CICLOS", "window_start"]],
        on=["cd_setor", "CICLOS"],
        validate="many_to_one",
    )


def _compare(daily: pd.DataFrame, actual: pd.DataFrame, target: str, as_of) -> pd.DataFrame:
    """Preserve missing predictions and treat forecasted days without orders as actual zero."""
    actual = actual.rename(columns={"window_start": "actual_window_start"})
    compared = daily.merge(actual, on=ACTUAL_KEYS, how="outer", validate="one_to_one")
    compared["actual"] = compared["actual"].fillna(0)
    compared["window_start"] = compared["window_start"].fillna(compared["actual_window_start"])
    compared["relative_day"] = (compared["date"] - compared["window_start"]).dt.days + 1
    compared["prediction_available"] = compared["items_pred"].notna()
    compared["error"] = compared["items_pred"] - compared["actual"]
    compared["abs_error"] = compared["error"].abs()
    return compared.assign(target_cycle=target, as_of=as_of).drop(columns="actual_window_start")


def _metrics(rows: pd.DataFrame) -> dict:
    """Report daily point errors and disclose actual demand lacking any model prediction."""
    covered = rows[rows["prediction_available"]]
    actual = covered["actual"]
    total = rows["actual"].sum()
    return {
        **_point_metrics(covered),
        "n_scored_days": len(covered),
        "n_unpredicted_rows": int((~rows["prediction_available"]).sum()),
        "unpredicted_actual_items": rows.loc[~rows["prediction_available"], "actual"].sum(),
        "actual_volume_coverage": actual.sum() / total if total else np.nan,
    }


def _point_metrics(covered: pd.DataFrame) -> dict:
    """Leave accuracy undefined for wholly uncovered cities instead of averaging empty arrays."""
    functions = {"mae": mae, "rmse": rmse, "wmape": wmape, "bias": bias, "bias_pct": bias_pct}
    if covered.empty:
        return dict.fromkeys(functions, np.nan)
    return {
        name: function(covered["actual"], covered["items_pred"])
        for name, function in functions.items()
    }


def summarize_validation(rows: pd.DataFrame) -> pd.DataFrame:
    """Report aggregate diagnostics and a separate summary for each replayed cycle."""
    groups = [
        dict(scope="cycle", CICLOS=cycle, **_metrics(group))
        for cycle, group in rows.groupby("target_cycle")
    ]
    return pd.DataFrame([dict(scope="overall", CICLOS="ALL", **_metrics(rows)), *groups])


def _validate_cycle(raw, calendar, target, history_cycles, level, shape):
    """Run the production predictor once before opening and only then score actual demand."""
    _require_complete_target(calendar, target)
    opening = calendar.loc[calendar["CICLOS"].eq(target), "opening_date"].iloc[0]
    as_of = opening - pd.Timedelta(days=1)
    history = prepare_histories(raw, as_of, history_cycles)
    windows, actual_windows = _replay_windows(raw, calendar, target, history)
    if windows.empty:
        raise ValueError(f"no sectors with a complete known forecast path for {target}")
    result = forecast_combined(raw, windows, as_of, history_cycles, level, shape)
    compared = _compare(result.daily, _actual_days(raw, actual_windows), target, as_of)
    compared["training_cycles"] = ",".join(result.training_cycles)
    return compared, windows.assign(as_of=as_of)


def _require_complete_target(calendar: pd.DataFrame, target: str) -> None:
    """Reject absent or incomplete targets rather than scoring truncated real demand."""
    usable = calendar.loc[calendar["is_complete"], "CICLOS"]
    if target not in set(usable):
        raise ValueError(f"validation cycle {target} is missing or incomplete")


def validate_combined(
    raw: pd.DataFrame,
    cycles: list[str],
    history_cycles: int,
    level_params: ForecastParams,
    shape_params: ShapeCandidate,
) -> ValidationResult:
    """Evaluate explicit held-out cycles with one fixed pair of model configurations."""
    _distinct_cycles(cycles)
    calendar = cycle_calendar(raw)
    folds = [
        _validate_cycle(raw, calendar, cycle, history_cycles, level_params, shape_params)
        for cycle in cycles
    ]
    rows = pd.concat([fold[0] for fold in folds], ignore_index=True)
    return ValidationResult(
        rows, summarize_validation(rows), pd.concat([fold[1] for fold in folds], ignore_index=True)
    )


def _distinct_cycles(cycles: list[str]) -> None:
    """Require an explicit nonempty list of unique holdout cycles."""
    if not cycles or len(set(cycles)) != len(cycles):
        raise ValueError("provide distinct validation_cycles")


def summarize_cities(rows: pd.DataFrame) -> pd.DataFrame:
    """Expose city-cycle errors without cancellation between cities."""
    keys = ["target_cycle", "cd_setor", "estado", "cidade"]
    return pd.DataFrame(
        [
            dict(zip(keys, identity, strict=True), **_metrics(group))
            for identity, group in rows.groupby(keys)
        ]
    )
