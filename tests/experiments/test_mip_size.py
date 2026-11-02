"""The closed-form MIP size must match the Pyomo model that is actually built."""

from __future__ import annotations

import pytest

from experiments.compare_modeling import run as compare_modeling
from experiments.mip_size.run import CYCLE_LENGTH_DAYS, size_row
from src.solver.mip.block_assignment import build_block_assignment_model


@pytest.mark.parametrize(("n_sectors", "n_cycles"), [(40, 1), (40, 3), (80, 2)])
def test_size_row_matches_built_model(n_sectors, n_cycles, monkeypatch):
    monkeypatch.setattr(
        compare_modeling, "CYCLE_LENGTH_RANGE", (CYCLE_LENGTH_DAYS, CYCLE_LENGTH_DAYS)
    )
    model = build_block_assignment_model(compare_modeling.generate_instance(n_sectors, n_cycles))
    row = size_row(n_sectors, n_cycles)

    assert row["variables"] == model.nvariables()
    assert row["constraints"] == model.nconstraints()
