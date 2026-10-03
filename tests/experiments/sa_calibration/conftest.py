"""Fixtures da calibração: a instância de 2 setores do teste do SA como CalibrationInstance."""

from __future__ import annotations

import pytest

from src.calibration.config import BudgetSettings
from src.calibration.instances import CalibrationInstance
from tests.solver.heuristics.test_simulated_annealing import _instance


@pytest.fixture
def sa_instance() -> dict:
    """Kwargs de `solve` da instância de 2 setores: ótimo com amplitude 0, As-Is com 20."""
    return _instance()


@pytest.fixture
def tiny_instance(sa_instance) -> CalibrationInstance:
    """A mesma instância no formato usado pela calibração (SA e MIP)."""
    return CalibrationInstance(
        name="tiny",
        sectors=sa_instance["sectors"],
        combinations=sa_instance["combinations"],
        days=sa_instance["days"],
        cd_sectors=sa_instance["cd_sectors"],
        daily_capacity=sa_instance["daily_capacity"],
        projected_demand=sa_instance["projected_demand"],
        current_assignment_sa=sa_instance["current_assignment"],
        current_assignment_mip={(s, d): 1 for s, d in sa_instance["current_assignment"].items()},
        max_churn=sa_instance["max_churn"],
        days_by_cycle=sa_instance["days_by_cycle"],
        level_strategy="naive_last",
        shape_strategy="plain_average",
    )


@pytest.fixture
def small_budget() -> BudgetSettings:
    """Orçamento curto para testes rápidos."""
    return BudgetSettings(max_iterations=500, trace_every=100)
