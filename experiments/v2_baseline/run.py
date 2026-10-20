"""Score the reference level candidates on base_tratada_v2, per cycle and per CD-day."""

from __future__ import annotations

import sys

from experiments.compare_forecasters.run import run

DEFAULT_CONFIG_PATH = "configs/experiments/v2_baseline/all.yaml"


if __name__ == "__main__":
    comparison = run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH)
    print()
    print(comparison.to_string(index=False))
    print()
    print(f"common subset: {int(comparison['n_scored_common'].iloc[0])} cycle points")
