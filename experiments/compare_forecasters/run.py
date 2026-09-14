"""Run every configured forecasting candidate through the shared harness and rank them fairly."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from pydantic import TypeAdapter

from src.config.schema import ForecastParams
from src.forecasting.candidates import arima, lightgbm, naive  # noqa: F401 - populate REGISTRY
from src.forecasting.comparison import compare
from src.forecasting.dataset import build_item_panel, load_demand_base
from src.forecasting.evaluation import EvaluationResult, RollingOriginSplit, evaluate
from src.forecasting.model import REGISTRY

DEFAULT_CONFIG_PATH = "configs/experiments/compare_forecasters/all.yaml"

_PARAMS_ADAPTER = TypeAdapter(ForecastParams)


def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _evaluate_all(config: dict, panel: pd.DataFrame) -> list[EvaluationResult]:
    split = RollingOriginSplit(**config["split"])
    results = []
    for entry in config["candidates"]:
        candidate = REGISTRY.build(_PARAMS_ADAPTER.validate_python(entry))
        print(f"  running {candidate.name} ...", flush=True)
        results.append(evaluate(candidate, panel, split))
    return results


def _errors_by_candidate(results: list[EvaluationResult]) -> pd.DataFrame:
    frames = [result.scored.assign(candidate=result.candidate_name) for result in results]
    return pd.concat(frames, ignore_index=True)


def run(config_path: str | Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    config = _load_config(config_path)

    raw = load_demand_base(config["paths"]["base_csv"])
    panel = build_item_panel(raw)
    print(
        f"[compare_forecasters] panel: {panel['cd_setor'].nunique()} sectors x "
        f"{panel['CICLOS'].nunique()} cycles"
    )

    results = _evaluate_all(config, panel)
    table = compare(results)

    _write(table, config["paths"]["output_csv"])
    _write(_errors_by_candidate(results), config["paths"]["errors_csv"])
    return table


def _write(frame: pd.DataFrame, path: str) -> None:
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
