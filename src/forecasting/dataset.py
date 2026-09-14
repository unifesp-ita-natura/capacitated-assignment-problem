"""Load the raw demand base; aggregate it into the per-(sector, cycle) panels forecasting uses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns the raw demand base must have. Anything else is ignored by the
# loader, so upstream schema changes that only add columns don't break it.
REQUIRED_COLUMNS = [
    "data_pedido",
    "cd_setor",
    "nm_ciclo",
    "aa_ciclo",
    "CICLOS",
    "total_itens_mascarado",
    "Dt Abertura",
    "Dt Fechamento",
    "Qtde dias",
    "dia_ciclo",
]


def load_demand_base(path: str | Path) -> pd.DataFrame:
    """Read the raw demand CSV and validate it has the columns forecasting needs."""
    raw = pd.read_csv(path, dtype={"cd_setor": str, "CICLOS": str})
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"demand base at {path} is missing required columns: {sorted(missing)}")

    for column in ["data_pedido", "Dt Abertura", "Dt Fechamento"]:
        raw[column] = pd.to_datetime(raw[column])
    return raw


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
    """Aggregate the raw base into one row per (sector, cycle): total items and cycle opening date.

    This is L_{s,k} in the paper's notation, in items (the unit the CD
    daily-capacity figures were given in — see docs/agent-log). Cycles the
    base only partially observed (per `cycle_calendar`) are dropped: keeping
    them would teach every forecasting candidate a demand collapse that is
    an artifact of the export window, not real. Days keep no block/sub-block
    information here on purpose — see `cycle_calendar`.
    """
    calendar = cycle_calendar(raw) if calendar is None else calendar
    complete_cycles = set(calendar.loc[calendar["is_complete"], "CICLOS"])

    panel = (
        raw[raw["CICLOS"].isin(complete_cycles)]
        .groupby(["cd_setor", "CICLOS"], as_index=False)
        .agg(items=("total_itens_mascarado", "sum"))
    )
    panel = panel.merge(calendar[["CICLOS", "opening_date"]], on="CICLOS", how="left")
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
