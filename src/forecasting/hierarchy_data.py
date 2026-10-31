"""Construct hierarchical node panels and historical rosters without future city identities."""

import json

import numpy as np
import pandas as pd

from src.forecasting.dataset import build_item_panel
from src.forecasting.hierarchy_config import level_keys

CITY_KEYS = ["cd_setor", "estado", "cidade"]
WINDOW_COLUMNS = ["CICLOS", "opening_date", "window_start", "cycle_days"]


def node_ids(frame: pd.DataFrame, level: str, city_grouping: str = "sector") -> pd.Series:
    """Encode identifiers without ambiguous separators or homonymous city collisions."""
    keys = pd.MultiIndex.from_frame(frame[level_keys(level, city_grouping)].astype(str))
    codes, unique = pd.factorize(keys, sort=False)
    names = [json.dumps([level, *key], ensure_ascii=False) for key in unique]
    return pd.Series(np.asarray(names, dtype=object)[codes], index=frame.index)


def validate_organization(panel: pd.DataFrame) -> None:
    """Require an unambiguous sector-to-management-to-region assignment per cycle."""
    _validate_dimensions(panel)
    _validate_counts(panel)
    if panel.groupby(["CICLOS", "cd_setor"])[["CD_RE", "CD_GV"]].nunique().gt(1).any().any():
        raise ValueError("each sector-cycle must belong to one management and region")


def _validate_dimensions(panel: pd.DataFrame) -> None:
    required = {*CITY_KEYS, "CD_RE", "CD_GV", *WINDOW_COLUMNS, "items"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"hierarchy inputs missing {sorted(missing)}")
    if panel[list(required)].isna().any().any():
        raise ValueError("hierarchy inputs contain missing identifiers, windows or demand")


def _validate_counts(panel: pd.DataFrame) -> None:
    if (panel["items"] < 0).any() or (panel["cycle_days"] <= 0).any():
        raise ValueError("demand must be nonnegative and cycle_days positive")


def _aggregate(
    panel: pd.DataFrame, level: str, parent: str | None, city_grouping: str
) -> pd.DataFrame:
    labeled = panel.assign(node=node_ids(panel, level, city_grouping))
    labeled["parent"] = "" if parent is None else node_ids(panel, parent, city_grouping)
    keys = ["node", "parent", "CICLOS"]
    totals = labeled.groupby(keys, as_index=False).agg(items=("items", "sum"))
    windows = labeled.drop_duplicates([*keys, "cd_setor"]).copy()
    windows["opening_weekday"] = windows["window_start"].dt.dayofweek
    windows["opening_day"] = windows["window_start"].dt.day
    geometry = windows.groupby(keys, as_index=False).agg(
        opening_date=("opening_date", "first"),
        closing_date=("closing_date", "max"),
        window_start=("window_start", "min"),
        cycle_days=("cycle_days", "mean"),
        days_min=("cycle_days", "min"),
        days_max=("cycle_days", "max"),
        opening_weekday_mean=("opening_weekday", "mean"),
        opening_day_mean=("opening_day", "mean"),
    )
    return totals.merge(geometry, on=keys).assign(level=level)


def _with_children(frame: pd.DataFrame, child: pd.DataFrame) -> pd.DataFrame:
    grouped = child.assign(active=child["items"].gt(0)).groupby(["parent", "CICLOS"])
    counts = (
        grouped.agg(
            n_children=("node", "nunique"),
            active_children=("active", "sum"),
            largest_child=("items", "max"),
        )
        .reset_index()
        .rename(columns={"parent": "node"})
    )
    result = frame.merge(counts, on=["node", "CICLOS"], how="left")
    result["largest_share"] = result["largest_child"] / result["items"].replace(0, np.nan)
    return result.drop(columns="largest_child")


def aggregate_nodes(
    panel: pd.DataFrame, levels: list[str], city_grouping: str = "sector"
) -> pd.DataFrame:
    """Sum cities once and describe scenario geometry without inventing a parent opening date."""
    validate_organization(panel)
    parents = [None, *levels[:-1]]
    frames = [
        _aggregate(panel, level, parent, city_grouping) for level, parent in zip(levels, parents)
    ]
    augmented = [_with_children(frame, child) for frame, child in zip(frames, frames[1:])]
    leaf = frames[-1].assign(n_children=0, active_children=0, largest_share=0)
    return pd.concat([*augmented, leaf], ignore_index=True)


def known_roster(past: pd.DataFrame) -> pd.DataFrame:
    """Carry latest known organization and every previously observed city into the next query."""
    cities = past[CITY_KEYS].drop_duplicates()
    organization = past.sort_values("opening_date").drop_duplicates("cd_setor", keep="last")
    return cities.merge(organization[["cd_setor", "CD_RE", "CD_GV"]], on="cd_setor")


def scenario_panel(roster: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Expand only planned sectors into known cities; no future purchases define the city roster."""
    if windows.duplicated("cd_setor").any():
        raise ValueError("one window per sector is required within a scenario-cycle")
    return roster.merge(windows, on="cd_setor", validate="many_to_one").assign(
        items=0.0, closing_date=pd.NaT
    )


def history_panel(raw: pd.DataFrame, calendar: pd.DataFrame, cycles: list[str]) -> pd.DataFrame:
    """Build the full past panel before cutting the window, retaining older known quiet cities."""
    past_cycles = calendar.loc[calendar["is_complete"], "CICLOS"]
    available = raw[raw["CICLOS"].isin(past_cycles)]
    panel = build_item_panel(available, calendar)
    panel = panel.merge(calendar[["CICLOS", "closing_date"]], on="CICLOS", validate="many_to_one")
    return panel[panel["CICLOS"].isin(cycles)].copy()
