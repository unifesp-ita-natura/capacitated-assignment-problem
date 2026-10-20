"""Tabulate and plot the MIP's variable and constraint counts across a sector/cycle sweep."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.mip_size.model_size import block_assignment_model_size
from src.visualizations.model_size import plot_model_size

OUTPUT_DIR = Path("experiments/mip_size/outputs")

N_COMBINATIONS = 15
N_CDS = 2
CYCLE_LENGTH_DAYS = 21

SECTOR_SWEEP = list(range(40, 1001, 40))
CYCLE_SWEEP = list(range(1, 21))


def size_row(n_sectors: int, n_cycles: int) -> dict[str, int]:
    """Counts for one instance, with every cycle `CYCLE_LENGTH_DAYS` long."""
    cycle_lengths = [CYCLE_LENGTH_DAYS] * n_cycles
    size = block_assignment_model_size(n_sectors, N_COMBINATIONS, N_CDS, cycle_lengths)
    return {
        "n_sectors": n_sectors,
        "n_cycles": n_cycles,
        "n_days": sum(cycle_lengths),
        "binary_variables": size.binary_variables,
        "variables": size.variables,
        "constraints": size.constraints,
    }


def run(
    sector_sweep: list[int] = SECTOR_SWEEP, cycle_sweep: list[int] = CYCLE_SWEEP
) -> pd.DataFrame:
    return pd.DataFrame(
        [size_row(n_sectors, n_cycles) for n_cycles in cycle_sweep for n_sectors in sector_sweep]
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sizes = run()
    sizes.to_csv(OUTPUT_DIR / "sizes.csv", index=False)
    plot_model_size(sizes).savefig(OUTPUT_DIR / "model_size.png", dpi=150, bbox_inches="tight")
    print(sizes.iloc[[0, -1]].to_string(index=False))
