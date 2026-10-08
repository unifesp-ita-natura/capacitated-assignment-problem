"""Run every configured forecasting candidate through the shared harness and rank them fairly."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from pydantic import TypeAdapter

import src.forecasting.candidates  # noqa: F401 - populate REGISTRY
from experiments.compare_forecasters.records import prepare_record, update_record
from src.config.schema import ForecastParams
from src.forecasting.comparison import compare
from src.forecasting.dataset import (
    CYCLE_KEYS,
    DailyBase,
    build_daily_base,
    build_item_panel,
    load_demand_base,
    sector_cycle_attributes,
)
from src.forecasting.evaluation import EvaluationResult, RollingOriginSplit, evaluate
from src.forecasting.model import REGISTRY

DEFAULT_CONFIG_PATH = "configs/experiments/compare_forecasters/all.yaml"

_PARAMS_ADAPTER = TypeAdapter(ForecastParams)


def _load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _evaluate_all(
    config: dict, panel: pd.DataFrame, daily_base: DailyBase | None = None
) -> list[EvaluationResult]:
    split = RollingOriginSplit(**config["split"])
    curve = config.get("daily", {}).get("curve", "sector")
    results = []
    for entry in config["candidates"]:
        candidate = REGISTRY.build(_PARAMS_ADAPTER.validate_python(entry))
        print(f"  running {candidate.name} ...", flush=True)
        results.append(evaluate(candidate, panel, split, daily_base, curve))
    return results


def _errors_by_candidate(results: list[EvaluationResult]) -> pd.DataFrame:
    frames = [result.scored.assign(candidate=result.candidate_name) for result in results]
    return pd.concat(frames, ignore_index=True)


def run(config_path: str | Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    config = _load_config(config_path)
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
        f"[compare_forecasters] panel: {panel['cd_setor'].nunique()} sectors x "
        f"{panel['CICLOS'].nunique()} cycles"
    )

    # A `daily:` block in the config turns on per-day scoring (see src/forecasting/daily.py).
    daily_base = build_daily_base(raw) if "daily" in config else None
    results = _evaluate_all(config, panel, daily_base)
    table = compare(results)

    # Region, sales management, CD and state ride along so results can be sliced by them.
    predictions = _errors_by_candidate(results).merge(
        sector_cycle_attributes(raw), on=CYCLE_KEYS, how="left"
    )
    return table, predictions


def _save_outputs(
    config: dict, directory: Path, table: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    """Archive results and retain explicitly configured legacy CSV destinations."""
    _write(table, directory / "comparison.csv")
    _write(predictions, directory / "predictions.csv")
    paths = config["paths"]
    for key, frame in (("output_csv", table), ("errors_csv", predictions)):
        if key in paths:
            _write(frame, paths[key])


def _write(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


if __name__ == "__main__":
    comparison = run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH)
    print()
    print(comparison.to_string(index=False))
    print()
    common = int(comparison["n_scored_common"].iloc[0])
    print(f"ranked by mae on the common subset ({common} points)")
