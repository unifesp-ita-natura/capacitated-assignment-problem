"""Validate scenario windows and prepare point-in-time histories for combined forecasts."""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from src.forecasting.dataset import build_item_panel, cycle_calendar
from src.forecasting.shape_daily import build_shape_days

WINDOW_KEYS = ["scenario_id", "cd_setor", "CICLOS"]
CITY_IDS = ["serie", "cd_setor", "estado", "cidade"]


class ScenarioWindow(BaseModel):
    """One sector's sales window within a complete scenario path."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    scenario_id: str = Field(min_length=1)
    cd_setor: str = Field(min_length=1)
    CICLOS: str = Field(pattern=r"^\d{6}$")
    opening_date: date
    window_start: date
    window_end: date
    emit: bool = True

    @model_validator(mode="after")
    def ordered_dates(self):
        if not self.opening_date <= self.window_start <= self.window_end:
            raise ValueError("require opening_date <= window_start <= window_end")
        return self


class Histories(NamedTuple):
    """Training panels and all previously known city identities."""

    level: pd.DataFrame
    shape: pd.DataFrame
    cities: pd.DataFrame
    cycles: list[str]
    hierarchy: pd.DataFrame


def prepare_windows(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse explicitly supplied future windows without inferring opening dates."""
    parsed = TypeAdapter(list[ScenarioWindow]).validate_python(frame.to_dict("records"))
    if not parsed:
        raise ValueError("provide at least one scenario window")
    windows = pd.DataFrame([row.model_dump() for row in parsed])
    for column in ["opening_date", "window_start", "window_end"]:
        windows[column] = pd.to_datetime(windows[column])
    windows["cycle_days"] = (windows["window_end"] - windows["window_start"]).dt.days + 1
    windows["n_days"] = windows["cycle_days"]
    _unique_windows(windows)
    return windows


def _unique_windows(windows: pd.DataFrame) -> None:
    """Reject duplicate windows and conflicting shared cycle openings."""
    if windows[WINDOW_KEYS].duplicated().any():
        raise ValueError("one window is required per scenario-sector-cycle")
    if windows.groupby(["scenario_id", "CICLOS"])["opening_date"].nunique().gt(1).any():
        raise ValueError("opening_date must be shared by all sectors of a cycle")


def validate_path(windows: pd.DataFrame, as_of: pd.Timestamp) -> None:
    """Require future emitted cycles and a complete sector roster at every forecast step."""
    targets = windows[windows["emit"]]
    if targets.empty or targets["opening_date"].min() <= as_of:
        raise ValueError("as_of must be before the first opening of every emitted cycle")
    sectors = windows.groupby("CICLOS")["cd_setor"].agg(frozenset)
    if sectors.nunique() != 1:
        raise ValueError("each cycle in a scenario path must list the same sectors")
    if windows.groupby("opening_date")["CICLOS"].nunique().gt(1).any():
        raise ValueError("distinct cycles must have distinct opening dates")


def available_calendar(raw: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep observations strictly before the declared forecast timestamp."""
    available = raw[raw["data_pedido"].lt(as_of)].copy()
    if available.empty:
        raise ValueError("no historical observations before as_of")
    calendar = cycle_calendar(available)
    calendar["is_complete"] &= calendar["closing_date"].lt(as_of.normalize())
    return available, calendar


def prepare_histories(raw: pd.DataFrame, as_of: pd.Timestamp, history_cycles: int) -> Histories:
    """Train only on complete closed cycles while retaining known zero-demand cities."""
    available, calendar = available_calendar(raw, as_of)
    complete = calendar[calendar["is_complete"]].sort_values(["opening_date", "CICLOS"])
    if len(complete) < history_cycles:
        raise ValueError(f"need {history_cycles} complete closed cycles; found {len(complete)}")
    cycles = complete["CICLOS"].tail(history_cycles).tolist()
    panel = build_item_panel(available, calendar=calendar)
    cities = panel[CITY_IDS].drop_duplicates()
    selected = available[available["CICLOS"].isin(cycles)]
    shape = build_shape_days(selected, calendar=complete[complete["CICLOS"].isin(cycles)])
    hierarchy = panel.merge(
        complete[["CICLOS", "closing_date"]], on="CICLOS", validate="many_to_one"
    )
    level = hierarchy[hierarchy["CICLOS"].isin(cycles)].copy()
    return Histories(level, shape, cities, cycles, hierarchy)


def validate_known_gaps(
    raw: pd.DataFrame, as_of: pd.Timestamp, history: Histories, windows: pd.DataFrame
) -> None:
    """Require explicit intermediate windows for known cycles not available for training."""
    _, calendar = available_calendar(raw, as_of)
    last = history.level["opening_date"].max()
    pending = calendar[
        calendar["opening_date"].gt(last)
        & calendar["opening_date"].le(windows["opening_date"].max())
    ]
    missing = set(pending["CICLOS"]) - set(windows["CICLOS"])
    if missing:
        raise ValueError(f"include intermediate cycle windows with emit=false: {sorted(missing)}")
