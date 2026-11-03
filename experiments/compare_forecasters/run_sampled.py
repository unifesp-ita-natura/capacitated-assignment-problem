"""Run the compare_forecasters harness on a seeded sample of sectors (config `sample` block)."""

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
    build_item_panel,
    load_demand_base,
    sector_cycle_attributes,
)
from src.forecasting.evaluation import RollingOriginSplit, evaluate
from src.forecasting.model import REGISTRY

_PARAMS_ADAPTER = TypeAdapter(ForecastParams)


def _load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sample_sectors(panel: pd.DataFrame, n_sectors: int, seed: int) -> pd.DataFrame:
    sectors = pd.Series(sorted(panel["cd_setor"].astype(str).unique()))
    if n_sectors >= len(sectors):
        return panel
    chosen = set(sectors.sample(n=n_sectors, random_state=seed))
    return panel[panel["cd_setor"].astype(str).isin(chosen)].reset_index(drop=True)


def _execute(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_demand_base(config["paths"]["base_csv"])
    needs_external = any(
        entry.get("model") == "lightgbm_external" for entry in config["candidates"]
    )
    panel = build_item_panel(raw, external=needs_external)

    sample = config.get("sample", {})
    panel = _sample_sectors(panel, sample.get("n_sectors", 10**9), sample.get("seed", 0))
    print(
        f"[compare_forecasters] sampled panel: {panel['serie'].nunique()} series "
        f"({panel['cd_setor'].nunique()} sectors) x {panel['CICLOS'].nunique()} cycles",
        flush=True,
    )

    split = RollingOriginSplit(**config["split"])
    results = []
    for entry in config["candidates"]:
        candidate = REGISTRY.build(_PARAMS_ADAPTER.validate_python(entry))
        print(f"  running {candidate.name} ...", flush=True)
        results.append(evaluate(candidate, panel, split))
    table = compare(results)

    frames = [result.scored.assign(candidate=result.candidate_name) for result in results]
    predictions = pd.concat(frames, ignore_index=True).merge(
        sector_cycle_attributes(raw), on=CYCLE_KEYS, how="left"
    )
    return table, predictions


def run(config_path: str | Path) -> pd.DataFrame:
    config = _load_config(config_path)
    directory, manifest = prepare_record(config_path, config)
    print(f"[compare_forecasters] run directory: {directory.resolve()}", flush=True)
    try:
        table, predictions = _execute(config)
        _write(table, directory / "comparison.csv")
        _write(predictions, directory / "predictions.csv")
        update_record(directory, manifest, "complete", n_predictions=len(predictions))
    except Exception as error:
        update_record(directory, manifest, "failed", error=str(error))
        raise
    return table


def _write(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


if __name__ == "__main__":
    comparison = run(sys.argv[1])
    print()
    print(comparison.to_string(index=False))
    print()
    common = int(comparison["n_scored_common"].iloc[0])
    print(f"ranked by mae on the common subset ({common} points)")
