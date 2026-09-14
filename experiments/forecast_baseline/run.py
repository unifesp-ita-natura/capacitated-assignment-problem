"""Run the naive candidate end-to-end through the shared harness against the real demand base."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from src.config.schema import NaiveParams
from src.forecasting.candidates import naive  # noqa: F401 - registers "naive" into REGISTRY
from src.forecasting.dataset import build_item_panel, load_demand_base
from src.forecasting.evaluation import EvaluationResult, RollingOriginSplit, evaluate
from src.forecasting.model import REGISTRY

DEFAULT_CONFIG_PATH = "configs/experiments/forecast_baseline/naive.yaml"


def _load_run_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(config_path: str | Path = DEFAULT_CONFIG_PATH) -> EvaluationResult:
    config = _load_run_config(config_path)

    raw = load_demand_base(config["paths"]["base_csv"])
    panel = build_item_panel(raw)

    candidate = REGISTRY.build(NaiveParams(**config["candidate"]))
    split = RollingOriginSplit(**config["split"])

    result = evaluate(candidate, panel, split)

    output_path = Path(config["paths"]["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.scored.to_csv(output_path, index=False)

    return result


def _print_report(result: EvaluationResult, output_path: str) -> None:
    summary = result.summary()
    print(f"[forecast_baseline] candidate={result.candidate_name}")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  wrote per-(sector,cycle) errors to {output_path}")
    if not result.missing.empty:
        n_sectors = result.missing["cd_setor"].nunique()
        n_missing = len(result.missing)
        print(f"  note: {n_missing} target(s) across {n_sectors} sector(s) had no prediction")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    run_result = run(config_path)
    _print_report(run_result, _load_run_config(config_path)["paths"]["output_csv"])
