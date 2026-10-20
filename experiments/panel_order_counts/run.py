"""Rank the order-count LightGBM against its own baseline, through the shared comparison driver."""

from __future__ import annotations

import sys

from experiments.compare_forecasters.run import run

DEFAULT_CONFIG_PATH = "configs/experiments/panel_order_counts/all.yaml"


if __name__ == "__main__":
    comparison = run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH)
    print()
    print(comparison.to_string(index=False))
    print()
    common = int(comparison["n_scored_common"].iloc[0])
    print(f"ranked by mae on the common subset ({common} points)")
