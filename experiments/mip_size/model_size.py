"""Closed-form variable and constraint counts of the block-assignment MIP, without building it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSize:
    """Size of the MIP built by `build_block_assignment_model`."""

    binary_variables: int
    continuous_variables: int
    constraints: int

    @property
    def variables(self) -> int:
        return self.binary_variables + self.continuous_variables


def block_assignment_model_size(
    n_sectors: int, n_combinations: int, n_cds: int, cycle_lengths: Sequence[int]
) -> ModelSize:
    """Counts mirror `_add_variables` and the `_add_*_constraint(s)` helpers one to one.

    Variables: `x[s, d]` (binary), `y[s, c]` (continuous share) plus `z_max[h]`,
    `z_min[h]` per cycle.
    Constraints: one assignment and one share-sums-to-one row per sector, a
    demand upper and lower bound per day, a capacity row per (CD, day), and
    the single churn row. Days are the sum of the cycle lengths, so the model
    is sized by the cycles' lengths and not only by how many there are.
    """
    n_days = sum(cycle_lengths)
    populated_cds = min(n_cds, n_sectors)
    return ModelSize(
        binary_variables=n_sectors * n_combinations,
        continuous_variables=n_sectors * n_cds + 2 * len(cycle_lengths),
        constraints=2 * n_sectors + 2 * n_days + populated_cds * n_days + 1,
    )
