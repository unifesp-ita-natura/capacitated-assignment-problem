"""Rank candidates per CD-day instead of per cycle, through the shared driver."""

from __future__ import annotations

import sys

from experiments.compare_forecasters.run import run

DEFAULT_CONFIG_PATH = "configs/experiments/daily_output/sector.yaml"


if __name__ == "__main__":
    comparison = run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH)
    print()
    print(comparison.to_string(index=False))
    print()
    common = int(comparison["n_scored_common"].iloc[0])
    print(f"ranked by mae per CD-day on the common subset ({common} cycle points)")
