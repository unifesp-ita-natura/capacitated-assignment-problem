"""Run every configured forecasting candidate through the shared harness and rank them fairly."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import src.forecasting.candidates  # noqa: F401 - populate REGISTRY
from src.forecasting.comparison import compare
from src.forecasting.dataset import (
    CYCLE_KEYS,
    DailyBase,
    build_daily_base,
    build_item_panel,
    load_demand_base,
    sector_cycle_attributes,
)
from src.forecasting.harness import (
    errors_by_candidate,
    evaluate_all,
    load_config,
    prepare_record,
    update_record,
    write_csv,
)

DEFAULT_CONFIG_PATH = "configs/experiments/compare_forecasters/all.yaml"


def run(config_path: str | Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    config = load_config(config_path)
    directory, manifest = prepare_record(config_path, config)
    print(f"[compare_forecasters] run directory: {directory.resolve()}", flush=True)
    try:
        table, predictions = _execute(config)
        _save_outputs(config, directory, table, predictions)
        update_record(directory, manifest, "complete", n_predictions=len(predictions))
    except Exception as error:
        update_record(directory, manifest, "failed", error=str(error))
        raise
    return table


def _execute(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_demand_base(config["paths"]["base_csv"])
    panel = build_item_panel(raw)
    print(
        f"[compare_forecasters] panel: {panel['serie'].nunique()} sector-city series "
        f"({panel['cd_setor'].nunique()} sectors) x {panel['CICLOS'].nunique()} cycles"
    )

    # A `daily:` block in the config turns on per-day scoring (see src/forecasting/daily.py).
    daily_base: DailyBase | None = build_daily_base(raw) if "daily" in config else None
    results = evaluate_all(config, panel, daily_base)
    table = compare(results)

    # Region, sales management and CD ride along so results can be sliced by them;
    # each series already carries its own state and city.
    predictions = errors_by_candidate(results).merge(
        sector_cycle_attributes(raw), on=CYCLE_KEYS, how="left"
    )
    return table, predictions


def _save_outputs(
    config: dict, directory: Path, table: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    """Archive results and retain explicitly configured legacy CSV destinations."""
    write_csv(table, directory / "comparison.csv")
    write_csv(predictions, directory / "predictions.csv")
    paths = config["paths"]
    for key, frame in (("output_csv", table), ("errors_csv", predictions)):
        if key in paths:
            write_csv(frame, paths[key])


if __name__ == "__main__":
    comparison = run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH)
    print()
    print(comparison.to_string(index=False))
    print()
    common = int(comparison["n_scored_common"].iloc[0])
    print(f"ranked by mae on the common subset ({common} points)")
