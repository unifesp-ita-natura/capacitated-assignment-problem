"""Load the raw demand base; aggregate it into the per-(sector, cycle) panels forecasting uses."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

CYCLE_KEYS = ["cd_setor", "CICLOS"]

# Columns the raw demand base must have. Anything else is ignored by the
# loader, so upstream schema changes that only add columns don't break it.
REQUIRED_COLUMNS = [
    "data_pedido",
    "cd_setor",
    "nm_ciclo",
    "aa_ciclo",
    "CICLOS",
    "total_pedidos_mascarado",
    "total_volumes_mascarado",
    "total_itens_mascarado",
    "Dt Abertura",
    "Dt Fechamento",
    "dia_ciclo",
]

# base_tratada_v2 repeats an order once per opening date its sector had in the
# cycle; the copies differ only in these columns (see `_drop_fanned_out_rows`).
_OPENING_DEPENDENT = ("Dt Abertura", "dia_ciclo", "Qtde dias")


def load_demand_base(path: str | Path) -> pd.DataFrame:
    """Read the raw demand CSV and validate it has the columns forecasting needs."""
    raw = pd.read_csv(path, dtype={"cd_setor": str, "CICLOS": str})
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"demand base at {path} is missing required columns: {sorted(missing)}")

    for column in ["data_pedido", "Dt Abertura", "Dt Fechamento"]:
        raw[column] = pd.to_datetime(raw[column])
    if "Qtde dias" not in raw.columns:  # dropped in base_tratada_v2
        raw["Qtde dias"] = (raw["Dt Fechamento"] - raw["Dt Abertura"]).dt.days + 1
    return _drop_fanned_out_rows(raw)


def _drop_fanned_out_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep one copy of an order that the base repeats under several opening dates.

    In base_tratada_v2 an order can appear once per opening date its sector
    had in that cycle (1,190 rows, 0.17% of items), identical apart from
    `Dt Abertura` and the columns derived from it. Counting every copy
    would double those items. The latest opening date is kept, so a sector's window (the earliest
    opening among its kept rows) starts where its first order was placed.
    """
    order_keys = [c for c in raw.columns if c not in _OPENING_DEPENDENT]
    latest_last = raw.sort_values("Dt Abertura", kind="stable")
    return latest_last.drop_duplicates(order_keys, keep="last").sort_index()


def cycle_calendar(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per cycle: its opening date, closing date, and whether the base fully covers it.

    The opening date is the earliest `Dt Abertura` across all blocks/sub-blocks
    in the cycle, not each sector's own block-specific opening date — block
    assignment is the optimizer's decision variable, so it must never leak
    into the series a forecast is dated by (see docs/agent-log for the
    decision). A cycle is "complete" only if its full opening-to-closing
    window falls inside the date range the base actually observed; a cycle
    whose closing date is still in the future relative to the base's last
    recorded day is a partial cycle, not a low-demand one.
    """
    base_start, base_end = raw["data_pedido"].min(), raw["data_pedido"].max()
    calendar = raw.groupby("CICLOS", as_index=False).agg(
        opening_date=("Dt Abertura", "min"),
        closing_date=("Dt Fechamento", "max"),
    )
    calendar["is_complete"] = (calendar["opening_date"] >= base_start) & (
        calendar["closing_date"] <= base_end
    )
    return calendar.sort_values("CICLOS").reset_index(drop=True)


def build_item_panel(raw: pd.DataFrame, calendar: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate the raw base into one row per (sector, cycle): items, orders, volumes, date.

    `items` is L_{s,k} in the paper's notation, in items (the unit the CD
    daily-capacity figures were given in — see docs/agent-log), and stays
    the forecast target. `orders` and `volumes` ride along as predictors,
    never as targets: a cycle's own order count isn't known when its volume
    is forecast, so only their lags are usable (see `features.py`). Cycles
    the base only partially observed (per `cycle_calendar`) are dropped:
    keeping them would teach every forecasting candidate a demand collapse
    that is an artifact of the export window, not real.

    `opening_date` is the cycle's, shared by every sector (see
    `cycle_calendar`), and orders the series. `window_start` and
    `cycle_days` are the sector's own window, the scenario inputs a
    candidate may use (see `model.forecast`).
    """
    calendar = cycle_calendar(raw) if calendar is None else calendar
    complete = raw[raw["CICLOS"].isin(set(calendar.loc[calendar["is_complete"], "CICLOS"]))]

    panel = complete.groupby(CYCLE_KEYS, as_index=False).agg(
        items=("total_itens_mascarado", "sum"),
        orders=("total_pedidos_mascarado", "sum"),
        volumes=("total_volumes_mascarado", "sum"),
    )
    panel = panel.merge(calendar[["CICLOS", "opening_date"]], on="CICLOS", how="left")
    panel = panel.merge(_sector_windows(complete), on=CYCLE_KEYS)
    return panel.sort_values(["cd_setor", "opening_date"]).reset_index(drop=True)


def build_shape_panel(raw: pd.DataFrame, calendar: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (sector, cycle, relative position): that day's share of the cycle's total items.

    Relative position is `dia_ciclo / Qtde dias`, a fraction of the sales
    window rather than a calendar day or a block-dependent offset, so a
    sector's shape stays comparable across cycles even when it migrates
    between blocks (see docs/agent-log). This panel is not consumed by the
    volume-forecasting harness in this module — it's the input the separate
    shape-forecasting work keys off.
    """
    calendar = cycle_calendar(raw) if calendar is None else calendar
    complete_cycles = set(calendar.loc[calendar["is_complete"], "CICLOS"])

    complete = raw[raw["CICLOS"].isin(complete_cycles)].copy()
    complete["relative_position"] = complete["dia_ciclo"] / complete["Qtde dias"]

    daily = complete.groupby(["cd_setor", "CICLOS", "relative_position"], as_index=False).agg(
        items=("total_itens_mascarado", "sum")
    )
    cycle_totals = daily.groupby(["cd_setor", "CICLOS"])["items"].transform("sum")
    daily["share"] = daily["items"] / cycle_totals
    return daily.sort_values(["cd_setor", "CICLOS", "relative_position"]).reset_index(drop=True)


class DailyBase(NamedTuple):
    """What the daily harness needs: real items per (sector, CD, cycle, date), and each window."""

    actuals: pd.DataFrame  # cd_setor, cd_cd, CICLOS, date, items — only days with an order
    windows: pd.DataFrame  # cd_setor, CICLOS, window_start, n_days — the sector's own window


def build_daily_base(raw: pd.DataFrame, calendar: pd.DataFrame | None = None) -> DailyBase:
    """Split the raw base into per-day actuals and per-(sector, cycle) windows (complete cycles).

    Days without orders have no row in `actuals`: the base only records days
    that sold something, so the zeros are recovered from `windows` (a sector's
    window is its own `Dt Abertura`..`Dt Fechamento`, see `_sector_windows`).
    """
    if "cd_cd" not in raw.columns:
        raise ValueError("the daily harness needs a `cd_cd` column in the demand base")
    calendar = cycle_calendar(raw) if calendar is None else calendar
    complete = raw[raw["CICLOS"].isin(set(calendar.loc[calendar["is_complete"], "CICLOS"]))]

    actuals = complete.groupby(["cd_setor", "cd_cd", "CICLOS", "data_pedido"], as_index=False).agg(
        items=("total_itens_mascarado", "sum")
    )
    return DailyBase(
        actuals=actuals.rename(columns={"data_pedido": "date"}),
        windows=_sector_windows(complete).rename(columns={"cycle_days": "n_days"}),
    )


def _sector_windows(raw: pd.DataFrame) -> pd.DataFrame:
    """Each (sector, cycle)'s own sales window: its first day and its length in days.

    This is where the block-dependent start lives. It is what a scenario
    query asks about ("what if this sector opened on day d for n days?"),
    never a stand-in for the block itself.
    """
    windows = raw.groupby(CYCLE_KEYS, as_index=False).agg(
        window_start=("Dt Abertura", "min"), window_end=("Dt Fechamento", "max")
    )
    windows["cycle_days"] = (windows["window_end"] - windows["window_start"]).dt.days + 1
    return windows.drop(columns="window_end")
