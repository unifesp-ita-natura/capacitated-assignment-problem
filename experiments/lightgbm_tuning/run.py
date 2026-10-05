"""Sweep the pooled model's capacity and regularization, then rank the grid on held-back folds."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.config.schema import LightGBMParams
from src.forecasting.candidates import lightgbm  # noqa: F401 - populates REGISTRY
from src.forecasting.dataset import build_item_panel, load_demand_base
from src.forecasting.evaluation import RollingOriginSplit, evaluate
from src.forecasting.model import REGISTRY

DEFAULT_CONFIG_PATH = "configs/experiments/lightgbm_tuning/all.yaml"


def _grid(config: dict) -> list[dict]:
    axes = config["grid"]
    names = sorted(axes)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(axes[n] for n in names))
    ]


def run(config_path: str | Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    panel = build_item_panel(load_demand_base(config["paths"]["base_csv"]))
    split = RollingOriginSplit(**config["split"])
    fixed = config["fixed"]

    rows = []
    points = _grid(config)
    for index, point in enumerate(points, start=1):
        params = LightGBMParams(**fixed, **point)
        result = evaluate(REGISTRY.build(params), panel, split)
        summary = result.summary()
        rows.append({**point, "mae": summary["mae"], "n_scored": summary["n_scored"]})
        print(f"  [{index}/{len(points)}] {point} -> {summary['mae']:.1f}", flush=True)

    table = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    output = Path(config["paths"]["output_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return table


if __name__ == "__main__":
    grid = run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH)
    print()
    print(grid.head(12).to_string(index=False))
