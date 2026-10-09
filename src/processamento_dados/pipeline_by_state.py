"""Builds demand-level/-shape CSVs split by (sector, state) from the real per-order demand base.

`pipeline.py`'s `demanda_level.csv`/`demanda_shape.csv` can't carry a state
column: they're built from the Simulador workbook's "Base Demanda" sheet,
which has no geography at all. The only source with a real, non-guessed
`estado` per row is the per-order demand export (`Unifesp_Demanda_v2.csv`);
this module joins that export to the Simulador workbook's calendar (for each
sector-cycle's open/close dates) and reuses the `group_by_state` panel
builders in `src/forecasting/dataset.py` to produce one row per (sector,
state, cycle) instead of one row per (sector, cycle).

A sector is a commercial grouping, not a geographic one (see
docs/context/natura-client-answers.md) — most sectors span several states —
so these outputs are a sibling of, not a replacement for, `pipeline.py`'s
files: nothing downstream of `demanda_level.csv`/`demanda_shape.csv` expects
a split sector, and the two pipelines pull from different raw exports (so
their totals won't reconcile row-for-row — see `docs/context/
demanda-level-shape.md`).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.forecasting.dataset import build_item_panel, build_shape_panel, load_demand_base
from src.processamento_dados.pipeline import (
    DEFAULT_PROCESSED_PATH,
    DEFAULT_RAW_PATH,
    DEFAULT_SIMULADOR_FILE,
    build_calendario,
    load_excel,
    validate_simulador_file,
    write_csv,
)

logger = logging.getLogger(__name__)

DEFAULT_RAW_DEMAND_FILE = DEFAULT_RAW_PATH / "Unifesp_Demanda_v2.csv"

DEMAND_BASE_OUTPUT_NAME = "demand_base_with_state.csv"
DEMANDA_LEVEL_BY_STATE_OUTPUT_NAME = "demanda_level_by_state.csv"
DEMANDA_SHAPE_BY_STATE_OUTPUT_NAME = "demanda_shape_by_state.csv"

# Renames the dataset.py panel builders' generic column names to match
# pipeline.py's demanda_level.csv/demanda_shape.csv naming, so both pipelines'
# outputs read the same way.
LEVEL_COLUMN_RENAME = {
    "CICLOS": "ciclo",
    "opening_date": "date",
    "cycle_days": "cycle_duration",
    "items": "total_itens",
    "orders": "total_pedidos",
    "volumes": "total_volumes",
}
SHAPE_COLUMN_RENAME = {
    "CICLOS": "ciclo",
    "relative_position": "relative_date",
    "items": "total_itens",
}


def build_raw_demand_with_state(raw_demand_file: str | Path) -> pd.DataFrame:
    """Read the per-order demand export, keeping its real per-row `estado`.

    `cd_setor` there is prefixed ("S3336"); stripped to match the Simulador
    workbook's plain-integer sector codes. `ciclo` is built the same way
    `pipeline.py.build_demand` does, so it joins to the same calendar.
    """
    raw = pd.read_csv(raw_demand_file, dtype={"cd_setor": str})
    raw["data_pedido"] = pd.to_datetime(raw["data_pedido"], format="%d/%m/%Y")
    raw["cd_setor"] = raw["cd_setor"].str.lstrip("S").astype(int)
    raw["ciclo"] = raw["aa_ciclo"].astype(str) + raw["nm_ciclo"].astype(str).str.zfill(2)
    return raw


def merge_with_calendar(raw: pd.DataFrame, df_calendario: pd.DataFrame) -> pd.DataFrame:
    """Join each order to its sector-cycle's open/close dates; derive its day-in-cycle.

    Orders from a cycle the calendar doesn't know about (e.g. the export's
    history reaches further back than the calendar's) are dropped, same as
    `pipeline.py.merge_demand_with_calendar`.
    """
    n_before = len(raw)
    df = raw.merge(df_calendario, left_on=["ciclo", "cd_setor"], right_on=["CICLOS", "COD SETOR"])
    unmatched = n_before - len(df)
    logger.info(
        "%d/%d demand rows unmatched to calendar (%.1f%%)",
        unmatched,
        n_before,
        unmatched / n_before * 100,
    )
    df["Dt Abertura"] = pd.to_datetime(df["Dt Abertura"])
    df["Dt Fechamento"] = pd.to_datetime(df["Dt Fechamento"])
    df["dia_ciclo"] = (df["data_pedido"] - df["Dt Abertura"]).dt.days + 1
    return df


def _split_sector_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Split a `group_by_state` panel's composite `cd_setor` ("<sector>|<estado>")
    into separate `cd_setor` and `estado` columns."""
    split = panel["cd_setor"].str.split("|", n=1, expand=True)
    return panel.assign(cd_setor=split[0].astype(int), estado=split[1])


def run_pipeline(
    raw_demand_file: Path,
    simulador_file: Path,
    output_path: Path,
) -> None:
    """Build the demand base (with state) and the by-(sector, state) level/shape CSVs."""
    validate_simulador_file(simulador_file)
    output_path.mkdir(parents=True, exist_ok=True)

    df_calendario = build_calendario(load_excel(simulador_file))
    raw = merge_with_calendar(build_raw_demand_with_state(raw_demand_file), df_calendario)
    demand_base_path = write_csv(raw, output_path, DEMAND_BASE_OUTPUT_NAME)

    base = load_demand_base(demand_base_path)

    level = _split_sector_state(build_item_panel(base, group_by_state=True))
    write_csv(
        level.rename(columns=LEVEL_COLUMN_RENAME), output_path, DEMANDA_LEVEL_BY_STATE_OUTPUT_NAME
    )

    shape = _split_sector_state(build_shape_panel(base, group_by_state=True))
    write_csv(
        shape.rename(columns=SHAPE_COLUMN_RENAME), output_path, DEMANDA_SHAPE_BY_STATE_OUTPUT_NAME
    )

    logger.info("By-state pipeline finished successfully")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the raw demand export, Simulador workbook, output dir."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-demand-file",
        type=Path,
        default=DEFAULT_RAW_DEMAND_FILE,
        help=f"Path to the per-order demand export (default: {DEFAULT_RAW_DEMAND_FILE})",
    )
    parser.add_argument(
        "--simulador-file",
        type=Path,
        default=DEFAULT_SIMULADOR_FILE,
        help=f"Simulador workbook path, for calendar dates (default: {DEFAULT_SIMULADOR_FILE})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_PROCESSED_PATH,
        help=f"Directory to write processed CSVs to (default: {DEFAULT_PROCESSED_PATH})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run_pipeline(args.raw_demand_file, args.simulador_file, args.output_path)
