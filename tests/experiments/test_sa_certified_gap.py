"""Tests for the certified-gap arithmetic of the sa_certified_gap experiment."""

import pytest

from experiments.sa_certified_gap.run import certified_gap


def test_gap_against_incumbent_and_optimum():
    gap = certified_gap(110.0, 100.0)
    assert gap["vs_incumbent"] == pytest.approx(10 / 110)
    assert gap["vs_optimum"] == pytest.approx(0.10)


def test_missing_or_zero_bound_gives_no_optimum_gap():
    assert certified_gap(5.0, None) == {"vs_incumbent": None, "vs_optimum": None}
    assert certified_gap(5.0, 0.0)["vs_optimum"] is None
