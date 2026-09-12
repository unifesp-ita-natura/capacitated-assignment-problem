"""Builds the processed calendar, demand, and demand-shape CSVs from the Simulador workbook."""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RAW_PATH = Path("../data/raw")
DEFAULT_PROCESSED_PATH = Path("../data/processed")
DEFAULT_SIMULADOR_FILE = DEFAULT_RAW_PATH / "Simulador Bloco e Subbloco_v2.xlsx"

EXCEL_ENGINE = "calamine"
CALENDARIO_SHEET = "Calendário FV"
DEMANDA_SHEET = "Base Demanda"

DEMAND_COLUMNS_NEEDED = [
    "data_pedido",
    "cd_setor",
    "cd_cd",
    "nm_ciclo",
    "aa_ciclo",
    "total_pedidos",
    "total_volumes",
    "total_itens",
]

CALENDARIO_OUTPUT_NAME = "calendario_fv.csv"
DEMANDA_WITH_CALENDAR_OUTPUT_NAME = "demanda_with_calendar.csv"
DEMANDA_LEVEL_OUTPUT_NAME = "demanda_level.csv"
DEMANDA_SHAPE_OUTPUT_NAME = "demanda_shape.csv"


def validate_simulador_file(simulador_file: Path) -> None:
    """Raise if the Simulador workbook is missing."""
    if not simulador_file.exists():
        raise FileNotFoundError(f"Missing Simulador file: {simulador_file}")


def load_excel(simulador_file: Path) -> pd.ExcelFile:
    """Open the Simulador workbook."""
    logger.info("Loading Simulador workbook from %s", simulador_file)
    return pd.ExcelFile(simulador_file, engine=EXCEL_ENGINE)


def build_calendario(excel: pd.ExcelFile) -> pd.DataFrame:
    """Extract and clean the calendar sheet (drops empty columns and the row index column)."""
    logger.info("Parsing sheet '%s'", CALENDARIO_SHEET)
    df_calendario = excel.parse(CALENDARIO_SHEET)
    df_calendario = df_calendario.dropna(how="all", axis=1)
    df_calendario = df_calendario.drop(columns=["#"])
    df_calendario["CICLOS"] = df_calendario["CICLOS"].astype(str)
    logger.info("Calendario shape: %s", df_calendario.shape)
    return df_calendario


def build_demand(excel: pd.ExcelFile) -> pd.DataFrame:
    """Extract the demand sheet, keeping only the needed columns and deriving the 'ciclo' key."""
    logger.info("Parsing sheet '%s'", DEMANDA_SHEET)
    df_demand = excel.parse(DEMANDA_SHEET)
    df_demand = df_demand[DEMAND_COLUMNS_NEEDED]
    df_demand["ciclo"] = df_demand["aa_ciclo"].astype(str) + df_demand["nm_ciclo"].astype(
        str
    ).str.zfill(2)
    logger.info("Demand shape: %s", df_demand.shape)
    return df_demand


def merge_demand_with_calendar(
    df_demand: pd.DataFrame, df_calendario: pd.DataFrame
) -> pd.DataFrame:
    """Join demand rows to their calendar entry by (ciclo, cd_setor), logging unmatched rows."""
    n_before = len(df_demand)
    df_check = df_demand.merge(
        df_calendario,
        left_on=["ciclo", "cd_setor"],
        right_on=["CICLOS", "COD SETOR"],
        how="left",
        indicator=True,
    )
    unmatched = df_check["_merge"].eq("left_only").sum()
    logger.info(
        "%d/%d demand rows unmatched to calendar (%.1f%%)",
        unmatched,
        n_before,
        unmatched / n_before * 100,
    )

    df = df_demand.merge(
        df_calendario, left_on=["ciclo", "cd_setor"], right_on=["CICLOS", "COD SETOR"]
    )
    df["Dt Abertura"] = pd.to_datetime(df["Dt Abertura"])
    df["Dt Fechamento"] = pd.to_datetime(df["Dt Fechamento"])
    logger.info("Demand with calendar shape: %s", df.shape)
    return df


def build_demand_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate demand to one row per (cd_setor, ciclo) with cycle totals and duration."""
    logger.info("Building sector-cycle level aggregation")
    df_level = (
        df.groupby(["cd_setor", "ciclo"], as_index=False)
        .agg(
            date=("Dt Abertura", "first"),
            cycle_duration=("Qtde dias", "first"),
            total_pedidos=("total_pedidos", "sum"),
            total_volumes=("total_volumes", "sum"),
            total_itens=("total_itens", "sum"),
        )
        .sort_values(by=["cd_setor", "date"])
        .reset_index(drop=True)
    )
    logger.info("Demand level shape: %s", df_level.shape)
    return df_level


def build_demand_shape(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the daily demand shape (share of cycle total per day) for each sector-cycle."""
    logger.info("Building demand shape")
    df_shape = df.copy()
    df_shape["data_pedido"] = pd.to_datetime(df_shape["data_pedido"])
    df_shape["Dt Abertura"] = pd.to_datetime(df_shape["Dt Abertura"])
    df_shape["relative_date"] = (df_shape["data_pedido"] - df_shape["Dt Abertura"]).dt.days / df[
        "Qtde dias"
    ]

    # exclude orders outside of cycle (don't know if that's the best way to deal with it)
    df_shape = df_shape[df_shape["relative_date"] <= 1]

    # daily aggregation: collapses multiple distribution centers / records on the same day for a
    # sector into single daily totals
    df_shape = df_shape.groupby(
        ["cd_setor", "ciclo", "data_pedido", "relative_date"],
        as_index=False,
    )[["total_pedidos", "total_volumes", "total_itens"]].sum()

    # extract cycle totals (items, orders, volume)
    df_shape[["ciclo_total_pedidos", "ciclo_total_volumes", "ciclo_total_itens"]] = (
        df_shape.groupby(["cd_setor", "ciclo"])[
            ["total_pedidos", "total_volumes", "total_itens"]
        ].transform("sum")
    )

    # share conversion: convert per day (items, orders, volume) to share of cycle total per sector
    df_shape["share_pedidos"] = df_shape["total_pedidos"] / df_shape["ciclo_total_pedidos"]
    df_shape["share_volumes"] = df_shape["total_volumes"] / df_shape["ciclo_total_volumes"]
    df_shape["share_itens"] = df_shape["total_itens"] / df_shape["ciclo_total_itens"]

    logger.info("Demand shape output shape: %s", df_shape.shape)
    return df_shape


def write_csv(df: pd.DataFrame, output_path: Path, name: str) -> Path:
    """Write a dataframe to CSV under output_path and return the file path."""
    output_file = output_path / name
    df.to_csv(output_file, index=False)
    logger.info("Wrote %s (%d rows)", output_file, len(df))
    return output_file


def run_pipeline(simulador_file: Path, output_path: Path) -> None:
    """Run the full processing pipeline: calendar, demand+calendar, demand level, demand shape."""
    validate_simulador_file(simulador_file)
    output_path.mkdir(parents=True, exist_ok=True)

    excel = load_excel(simulador_file)

    df_calendario = build_calendario(excel)
    write_csv(df_calendario, output_path, CALENDARIO_OUTPUT_NAME)

    df_demand = build_demand(excel)
    df = merge_demand_with_calendar(df_demand, df_calendario)
    write_csv(df, output_path, DEMANDA_WITH_CALENDAR_OUTPUT_NAME)

    df_level = build_demand_level(df)
    write_csv(df_level, output_path, DEMANDA_LEVEL_OUTPUT_NAME)

    df_shape = build_demand_shape(df)
    write_csv(df_shape, output_path, DEMANDA_SHAPE_OUTPUT_NAME)

    logger.info("Pipeline finished successfully")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the input Simulador file and output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulador-file",
        type=Path,
        default=DEFAULT_SIMULADOR_FILE,
        help=f"Path to the Simulador workbook (default: {DEFAULT_SIMULADOR_FILE})",
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
    run_pipeline(args.simulador_file, args.output_path)
