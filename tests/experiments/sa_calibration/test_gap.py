"""Testes do gap contra o MIP e da escolha da referência (ótimo x limite inferior)."""

from __future__ import annotations

import math

import pytest

from src.calibration.gap import INFEASIBLE_GAP, compute_gap, mip_reference


def test_gap_is_relative_to_the_reference():
    assert compute_gap(110.0, True, 100.0) == pytest.approx(10.0)


def test_gap_can_be_negative_within_the_mip_tolerance():
    assert compute_gap(99.5, True, 100.0) == pytest.approx(-0.5)


def test_infeasible_run_gets_the_infeasible_gap():
    assert compute_gap(1.0, False, 100.0) == INFEASIBLE_GAP


@pytest.mark.parametrize("reference", [None, 0.0])
def test_gap_is_nan_without_a_usable_reference(reference):
    assert math.isnan(compute_gap(5.0, True, reference))


def test_closed_mip_uses_the_incumbent():
    record = {
        "instance": "a",
        "termination": "optimal",
        "lower_bound": 98.0,
        "upper_bound": 100.0,
        "gap": 0.02,
    }

    ref = mip_reference(record)

    assert (ref.value, ref.kind, ref.mip_gap, ref.incumbent) == (100.0, "optimal", 0.02, 100.0)


def test_open_mip_uses_the_lower_bound():
    record = {
        "instance": "a",
        "termination": "maxTimeLimit",
        "lower_bound": 60.0,
        "upper_bound": 100.0,
        "gap": 0.4,
    }

    ref = mip_reference(record)

    assert (ref.value, ref.kind, ref.incumbent) == (60.0, "lower_bound", 100.0)


def test_missing_mip_record_has_no_reference():
    ref = mip_reference({"instance": "a"})

    assert (ref.value, ref.kind) == (None, "missing")
