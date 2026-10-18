"""Testes da montagem de instâncias da calibração a partir do gerador e do forecasting."""

from __future__ import annotations

import pytest

from src.calibration.config import InstanceSpec
from src.calibration.instances import ForecastChoice, build_instance
from src.solver.inputs import ProblemInstance

SPEC = InstanceSpec(n_sectors=6, n_cycles_horizon=2, seed=1)
FORCED_NAMES = ("naive_last", "plain_average")
FORCED = ForecastChoice(level=FORCED_NAMES[0], shape=FORCED_NAMES[1])


@pytest.fixture(scope="module")
def instance():
    return build_instance("small", SPEC, FORCED)


def test_to_instance_builds_the_shared_problem_instance(instance):
    assert isinstance(instance.to_instance(), ProblemInstance)


def test_forced_strategies_are_recorded(instance):
    metadata = instance.metadata()

    assert (metadata["level_strategy"], metadata["shape_strategy"]) == FORCED_NAMES
    assert metadata["n_sectors"] == SPEC.n_sectors
    assert metadata["n_cycles_horizon"] == SPEC.n_cycles_horizon


def test_same_spec_builds_the_same_instance(instance):
    again = build_instance("small", SPEC, FORCED)

    assert again.projected_demand == instance.projected_demand
    assert again.current_assignment == instance.current_assignment


def test_capacity_multiplier_scales_capacity():
    loose = build_instance("a", SPEC, FORCED)
    tight = build_instance("b", SPEC.model_copy(update={"capacity_multiplier": 0.5}), FORCED)

    key = next(iter(loose.daily_capacity))
    assert tight.daily_capacity[key] == pytest.approx(0.5 * loose.daily_capacity[key])
    assert tight.max_churn == loose.max_churn == SPEC.max_churn * SPEC.n_sectors
