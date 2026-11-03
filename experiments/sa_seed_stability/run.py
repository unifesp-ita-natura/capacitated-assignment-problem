"""Driver do sa_seed_stability: re-resolve o SA com várias seeds e cenários de previsão."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from pydantic import TypeAdapter

from src.config import RunConfig, load_config
from src.config.schema import CombinedLevelParams
from src.forecasting.dataset import load_demand_base
from src.main import _build_instance
from src.pipeline.forecast_stage import forecast_horizon, plan_horizon
from src.pipeline.realized import realized_instance
from src.pipeline.stability import SeedOutcome, run_seed, spread

DEFAULT_CONFIG = "configs/experiments/sa_seed_stability/full.yaml"
KEYS = ["scenario", "seed"]
METRICS = [
    "forecast_to_be",
    "realized_to_be",
    "forecast_improvement_pct",
    "realized_improvement_pct",
]


def _record(scenario: str, outcome: SeedOutcome) -> dict:
    return {
        "scenario": scenario,
        **vars(outcome),
        "forecast_improvement_pct": outcome.forecast_improvement_pct,
        "realized_improvement_pct": outcome.realized_improvement_pct,
    }


def _done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    return {(r.scenario, int(r.seed)) for r in pd.read_csv(path).itertuples()}


def _append(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _run_scenario(name: str, level, base: RunConfig, raw, seeds, path: Path) -> None:
    pipeline = base.pipeline
    plan = plan_horizon(raw, pipeline.horizon_cycles, pipeline.history_cycles)
    forecast = forecast_horizon(raw, plan, pipeline, level)
    cycles = forecast.training_cycles
    real = _build_instance(raw, plan, base, forecast.daily, cycles)
    predicted = real.instance
    actual = realized_instance(predicted, real, raw, plan.targets)
    todo = [s for s in seeds if (name, s) not in _done(path)]
    print(f"[{name}] {len(seeds) - len(todo)} done, {len(todo)} to run", flush=True)
    for seed in todo:
        outcome = run_seed(seed, predicted, actual)
        _append(path, _record(name, outcome))
        print(
            f"[{name}] seed {seed}: forecast {outcome.forecast_improvement_pct:.1f}% "
            f"realized {outcome.realized_improvement_pct:.1f}% ({outcome.seconds:.0f}s)",
            flush=True,
        )


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Spread of each metric per scenario and pooled over every scenario."""
    rows = []
    groups = [*runs.groupby("scenario"), ("ALL", runs)]
    for scenario, frame in groups:
        for metric in METRICS:
            rows.append({"scenario": scenario, "metric": metric, **spread(frame[metric])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.config).read_text())
    out = Path(spec["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "runs.csv"
    if args.overwrite and path.exists():
        path.unlink()
    base = load_config(spec["base_config"])
    raw = load_demand_base(base.pipeline.base_csv)
    adapter = TypeAdapter(CombinedLevelParams)
    for name, params in spec["scenarios"].items():
        _run_scenario(name, adapter.validate_python(params), base, raw, spec["seeds"], path)
    summary = summarize(pd.read_csv(path))
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
