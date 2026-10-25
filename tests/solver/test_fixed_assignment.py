"""Tests that pinned sectors (`fixed_assignment`) are honored by every solver family."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
from pydantic import ValidationError

from src.cli.main import TOY_INSTANCE
from src.config.schema import PipelineConfig, RunConfig
from src.pipeline.solve_stage import SOLVERS, run_solver
from src.solver.heuristics import simulated_annealing as sa
from src.solver.mip.block_assignment import build_block_assignment_model
from src.solver.real_inputs import validate_fixed_churn, validate_fixed_sectors

PINNED = replace(TOY_INSTANCE, max_churn=2, fixed_assignment={1: 2})


def _config(family: str) -> RunConfig:
    return RunConfig(solver="appsi_highs", pipeline=PipelineConfig(solver_family=family))


def test_mip_model_fixes_the_pinned_sector_variables() -> None:
    model = build_block_assignment_model(PINNED)
    assert model.x[1, 2].fixed and model.x[1, 2].value == 1
    assert model.x[1, 1].fixed and model.x[1, 1].value == 0
    assert not model.x[2, 1].fixed


@pytest.mark.parametrize("family", sorted(SOLVERS))
def test_every_solver_keeps_the_pinned_sector(family: str) -> None:
    solved = run_solver(family, PINNED, _config(family))
    assert solved.assignment is not None
    assert solved.assignment[1] == 2


def test_free_sector_rebalances_around_the_pin() -> None:
    solved = run_solver("mip", PINNED, _config("mip"))
    assert solved.assignment == {1: 2, 2: 1}
    assert solved.result.objective == pytest.approx(0.0)


def test_pinned_load_shrinks_the_capacity_left_to_free_sectors() -> None:
    tight = replace(PINNED, fixed_assignment={1: 1}, daily_capacity={(1, 1): 15.0, (1, 2): 100.0})
    solved = run_solver("mip", tight, _config("mip"))
    assert solved.assignment == {1: 1, 2: 2}


def test_sa_never_moves_pinned_sectors() -> None:
    result = sa.solve(PINNED, rng=random.Random(0))
    assert result.assignment[1] == 2


def test_sa_with_every_sector_pinned_returns_the_pins() -> None:
    everything = replace(PINNED, fixed_assignment={1: 2, 2: 2})
    assert sa.solve(everything, rng=random.Random(0)).assignment == {1: 2, 2: 2}


def test_validate_fixed_sectors_rejects_unknown_sector() -> None:
    with pytest.raises(ValueError, match="unknown sectors"):
        validate_fixed_sectors({"Z": 1}, ["A", "B"])


def test_validate_fixed_sectors_rejects_combination_out_of_range() -> None:
    with pytest.raises(ValueError, match="outside"):
        validate_fixed_sectors({"A": 16}, ["A"])


def test_validate_fixed_churn_rejects_pins_over_the_budget() -> None:
    with pytest.raises(ValueError, match="churn budget"):
        validate_fixed_churn({1: 2, 2: 1}, {1: 1, 2: 2}, max_churn=1)


def test_validate_fixed_churn_accepts_pins_matching_the_as_is() -> None:
    validate_fixed_churn({1: 1, 2: 2}, {1: 1, 2: 2}, max_churn=0)


def test_pipeline_config_rejects_non_positive_combination() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(fixed_sectors={"A": 0})
