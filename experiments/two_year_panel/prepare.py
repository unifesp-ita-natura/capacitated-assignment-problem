"""Build a two-year demand CSV from the client's v2 workbook and its 2024-2026 cycle calendar."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = "configs/experiments/two_year_panel/all.yaml"

# The v2 workbook has no cycle calendar of its own and the 2024-2025
# calendar extract has no block/sub-block, which is fine here: the
# forecasting panel is block-independent by design (see dataset.py).
_DEMAND_COLUMNS = {
    "cd_setor": "cd_setor",
    "nm_ciclo": "nm_ciclo",
    "aa_ciclo": "aa_ciclo",
    "total_pedidos_mascarado": "total_pedidos_mascarado",
    "total_volumes_mascarado": "total_volumes_mascarado",
    "total_itens_mascarado": "total_itens_mascarado",
}


def _read(path: str) -> pd.DataFrame:
    # calamine is already a project dependency and reads these 400k-row
    # sheets in seconds, where openpyxl takes minutes.
    return pd.read_excel(path, engine="calamine")


def _sector_id(column: pd.Series) -> pd.Series:
    """Strip the `S` prefix the v2 workbook uses, so ids match the calendar's numeric ones."""
    return column.astype(str).str.lstrip("S").str.strip()


def _cycle_id(frame: pd.DataFrame) -> pd.Series:
    year = frame["aa_ciclo"].astype(int).astype(str)
    number = frame["nm_ciclo"].astype(int).astype(str).str.zfill(2)
    return year + number


def _complete_rows(raw: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop rows the workbook left blank in a key column, before anything is cast to int.

    The v2 export carries a handful of rows with no cycle or no sector.
    Casting those to int raises rather than silently producing a bogus id,
    which is the behaviour we want — so they are removed here, explicitly,
    and the count is reported.
    """
    complete = raw.dropna(subset=columns)
    dropped = len(raw) - len(complete)
    if dropped:
        print(f"[two_year_panel] dropped {dropped:,} rows missing one of {columns}")
    return complete


def load_demand(path: str) -> pd.DataFrame:
    raw = _complete_rows(
        _read(path).rename(columns=_DEMAND_COLUMNS),
        ["data_pedido", "cd_setor", "nm_ciclo", "aa_ciclo", "total_itens_mascarado"],
    )
    raw["data_pedido"] = pd.to_datetime(raw["data_pedido"])
    raw["cd_setor"] = _sector_id(raw["cd_setor"])
    raw["CICLOS"] = _cycle_id(raw)
    return raw


def load_calendar(path: str) -> pd.DataFrame:
    """One row per cycle: its earliest opening and latest closing across every sector.

    Taking the extremes across sectors rather than each sector's own dates
    is the same rule `dataset.cycle_calendar` applies, and for the same
    reason: a sector's own opening date is set by its block, and block
    assignment is what the optimizer decides.
    """
    raw = _complete_rows(_read(path), ["CICLOS", "DT_ABERTURA", "DT_FECHAMENTO"])
    raw["CICLOS"] = raw["CICLOS"].astype(float).astype(int).astype(str)
    calendar = raw.groupby("CICLOS", as_index=False).agg(
        **{
            "Dt Abertura": ("DT_ABERTURA", "min"),
            "Dt Fechamento": ("DT_FECHAMENTO", "max"),
        }
    )
    for column in ["Dt Abertura", "Dt Fechamento"]:
        calendar[column] = pd.to_datetime(calendar[column]).dt.tz_localize(None)
    return calendar


def build(demand_path: str, calendar_path: str) -> pd.DataFrame:
    """Join demand to its cycle calendar and derive the columns `load_demand_base` requires."""
    demand = load_demand(demand_path)
    calendar = load_calendar(calendar_path)

    joined = demand.merge(calendar, on="CICLOS", how="inner")
    joined["Qtde dias"] = (joined["Dt Fechamento"] - joined["Dt Abertura"]).dt.days + 1
    joined["dia_ciclo"] = (joined["data_pedido"] - joined["Dt Abertura"]).dt.days + 1
    return joined[
        [
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
            "Qtde dias",
            "dia_ciclo",
        ]
    ]


def run(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    with open(config_path) as f:
        paths = yaml.safe_load(f)["paths"]

    frame = build(paths["demand_xlsx"], paths["calendar_xlsx"])
    output = Path(paths["base_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(
        f"[two_year_panel] {len(frame):,} rows, "
        f"{frame['data_pedido'].min().date()} to {frame['data_pedido'].max().date()}, "
        f"{frame['CICLOS'].nunique()} cycles, {frame['cd_setor'].nunique()} sectors -> {output}"
    )
    return output


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH)
