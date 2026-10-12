"""Joins calendar and macro-indicator external features onto the by-state demand panels."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.processamento_dados.external_calendar import (
    build_daily_calendar_features,
    summarize_cycle_calendar_features,
)
from src.processamento_dados.external_macro import fetch_macro_indicators
from src.processamento_dados.pipeline import DEFAULT_PROCESSED_PATH, write_csv

logger = logging.getLogger(__name__)

LEVEL_INPUT_NAME = "demanda_level_by_state.csv"
SHAPE_INPUT_NAME = "demanda_shape_by_state.csv"
LEVEL_OUTPUT_NAME = "demanda_level_by_state_with_external.csv"
SHAPE_OUTPUT_NAME = "demanda_shape_by_state_with_external.csv"

# How many quarterly SIDRA periods back to request — wide enough to cover the by-state
# files' observed window (2025-12 through 2026-07, i.e. quarters 202504..202603) with
# margin, even though the most recent quarter may not be published yet (see
# `external_macro.py`'s module docstring on PNAD Contínua's publication lag).
MACRO_LOOKBACK_PERIODS = "last 8"


def _quarter_period(d: pd.Timestamp) -> str:
    """Maps a date to its SIDRA quarterly period code (e.g. 2026-05 -> "202602")."""
    quarter = (d.month - 1) // 3 + 1
    return f"{d.year}{quarter:02d}"


def load_level(path: Path) -> pd.DataFrame:
    """Reads `demanda_level_by_state.csv`, parsing `date` as a real date column."""
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_shape(path: Path) -> pd.DataFrame:
    """Reads `demanda_shape_by_state.csv` as-is (no date column to parse yet)."""
    return pd.read_csv(path)


def _cycle_calendar_row(row: pd.Series) -> dict[str, int | bool]:
    """Builds one cycle's calendar rollup, from a `demanda_level_by_state.csv` row."""
    return summarize_cycle_calendar_features(row["estado"], row["date"], row["cycle_duration"])


def add_calendar_features_to_level(level: pd.DataFrame) -> pd.DataFrame:
    """Appends per-cycle calendar rollup columns (holiday counts, paydays) to the level panel."""
    rollups = pd.DataFrame(
        [_cycle_calendar_row(row) for _, row in level.iterrows()], index=level.index
    )
    return pd.concat([level, rollups], axis=1)


def add_quarter_period_to_level(level: pd.DataFrame) -> pd.DataFrame:
    """Adds a `periodo` column (SIDRA quarterly code) derived from the cycle's opening date."""
    return level.assign(periodo=level["date"].map(_quarter_period))


def _shape_calendar_date(shape: pd.DataFrame, level: pd.DataFrame) -> pd.Series:
    """Reconstructs each shape row's real calendar date from its cycle's opening date.

    `demanda_shape_by_state.csv` only carries `relative_date` (a [0, 1] fraction of the
    cycle), not a calendar date. The day offset within the cycle is
    `round(relative_date * cycle_duration)`, reversing how `relative_date` was built
    (`dia_ciclo / Qtde dias`, `dia_ciclo` being a 1-indexed day count) — see
    `src/forecasting/dataset.py::build_shape_panel`.

    Caveat: ~4.75% of `demanda_shape_by_state.csv` rows have `relative_date` outside
    [0, 1] (observed range -3.29 to 6.14), meaning the reconstructed date would fall
    outside that row's own cycle window. This isn't a bug in this formula — it
    reflects the by-state pipeline's `cycle_duration`/`date` being computed per
    (sector, estado) subgroup while the underlying order dates aren't, so a few
    orders land outside the subgroup's own nominal window. By default, callers
    (`add_calendar_features_to_shape`) drop these rows before calling this function
    (see `drop_out_of_cycle_window_rows`); pass `drop_out_of_cycle_window=False` to
    keep them as-computed instead.
    """
    join_key = ["cd_setor", "ciclo", "estado"]
    merged = shape.merge(level[[*join_key, "date", "cycle_duration"]], on=join_key, how="left")
    day_offset = (merged["relative_date"] * merged["cycle_duration"]).round().astype(int) - 1
    return merged["date"] + pd.to_timedelta(day_offset, unit="D")


def _out_of_cycle_window_mask(shape: pd.DataFrame) -> pd.Series:
    """Flags rows whose `relative_date` falls outside [0, 1] (see `_shape_calendar_date`)."""
    return (shape["relative_date"] < 0) | (shape["relative_date"] > 1)


def drop_out_of_cycle_window_rows(shape: pd.DataFrame) -> pd.DataFrame:
    """Drops rows whose reconstructed date would fall outside their own cycle window."""
    mask = _out_of_cycle_window_mask(shape)
    if mask.any():
        logger.warning(
            "Dropping %d/%d shape rows with relative_date outside [0, 1]; their "
            "reconstructed data_pedido would fall outside the row's own cycle window "
            "(see _shape_calendar_date's docstring)",
            mask.sum(),
            len(shape),
        )
    return shape.loc[~mask].reset_index(drop=True)


def _log_out_of_cycle_window_rows(shape: pd.DataFrame) -> None:
    """Logs how many rows have `relative_date` outside [0, 1], without dropping them."""
    mask = _out_of_cycle_window_mask(shape)
    if mask.any():
        logger.warning(
            "%d/%d shape rows have relative_date outside [0, 1]; their reconstructed "
            "data_pedido falls outside the row's own cycle window (see "
            "_shape_calendar_date's docstring)",
            mask.sum(),
            len(shape),
        )


def add_calendar_features_to_shape(
    shape: pd.DataFrame, level: pd.DataFrame, drop_out_of_cycle_window: bool = True
) -> pd.DataFrame:
    """Appends per-day calendar feature columns to the shape panel, keyed by reconstructed date.

    By default (`drop_out_of_cycle_window=True`), rows whose reconstructed date would
    fall outside their own cycle window are dropped first (see
    `drop_out_of_cycle_window_rows`). Pass `False` to keep them, as-computed, instead.
    """
    if drop_out_of_cycle_window:
        shape = drop_out_of_cycle_window_rows(shape)
    else:
        _log_out_of_cycle_window_rows(shape)

    with_date = shape.assign(data_pedido=_shape_calendar_date(shape, level))
    daily_features = build_daily_calendar_features(
        with_date["data_pedido"].unique(), with_date["estado"].unique()
    ).rename(columns={"date": "data_pedido"})
    return with_date.merge(daily_features, on=["data_pedido", "estado"], how="left")


def add_quarter_period_to_shape(shape_with_date: pd.DataFrame) -> pd.DataFrame:
    """Adds a `periodo` column (SIDRA quarterly code) derived from each row's calendar date."""
    return shape_with_date.assign(periodo=shape_with_date["data_pedido"].map(_quarter_period))


def _macro_wide(estados: list[str]) -> pd.DataFrame:
    """Fetches both macro indicators and pivots them into one row per (estado, periodo)."""
    long_form = fetch_macro_indicators(estados, MACRO_LOOKBACK_PERIODS)
    return long_form.pivot(
        index=["estado", "periodo"], columns="indicador", values="valor"
    ).reset_index()


def add_macro_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Left-joins macro indicators onto a frame that already has `estado`/`periodo` columns.

    Rows whose `periodo` hasn't been published yet by PNAD Contínua (see
    `external_macro.py`) get `NaN` for the macro columns rather than a fabricated value.
    """
    estados = sorted(frame["estado"].unique())
    macro = _macro_wide(estados)
    return frame.merge(macro, on=["estado", "periodo"], how="left")


def build_level_with_external(level: pd.DataFrame) -> pd.DataFrame:
    """Runs the full level-panel join: calendar rollup + quarterly macro indicators."""
    with_calendar = add_calendar_features_to_level(level)
    with_period = add_quarter_period_to_level(with_calendar)
    return add_macro_features(with_period)


def build_shape_with_external(
    shape: pd.DataFrame, level: pd.DataFrame, drop_out_of_cycle_window: bool = True
) -> pd.DataFrame:
    """Runs the full shape-panel join: daily calendar features + quarterly macro indicators."""
    with_calendar = add_calendar_features_to_shape(shape, level, drop_out_of_cycle_window)
    with_period = add_quarter_period_to_shape(with_calendar)
    return add_macro_features(with_period)


def run(
    processed_path: Path = DEFAULT_PROCESSED_PATH, drop_out_of_cycle_window: bool = True
) -> None:
    """Reads the by-state panels, joins external features, and writes the enriched siblings."""
    level = load_level(processed_path / LEVEL_INPUT_NAME)
    shape = load_shape(processed_path / SHAPE_INPUT_NAME)

    level_with_external = build_level_with_external(level)
    shape_with_external = build_shape_with_external(shape, level, drop_out_of_cycle_window)

    write_csv(level_with_external, processed_path, LEVEL_OUTPUT_NAME)
    write_csv(shape_with_external, processed_path, SHAPE_OUTPUT_NAME)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-path", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument(
        "--keep-out-of-cycle-window",
        dest="drop_out_of_cycle_window",
        action="store_false",
        help="Keep shape rows whose relative_date falls outside [0, 1] instead of dropping them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    run(args.processed_path, args.drop_out_of_cycle_window)
