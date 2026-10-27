"""Tests for the TSB and ADIDA intermittent-demand candidates."""

from __future__ import annotations

import numpy as np
import pytest

import src.forecasting.candidates  # noqa: F401
from src.config.schema import IntermittentParams
from src.forecasting.candidates.intermittent import adida, tsb
from src.forecasting.model import REGISTRY
from tests.forecasting.candidates._panels import panel, split


def test_tsb_on_a_series_that_always_buys_is_its_smoothed_size():
    assert tsb(np.array([100.0, 100.0, 100.0]), 0.1, 0.1) == pytest.approx(100.0)


def test_tsb_decays_when_a_series_stops_buying():
    buying, stopped = np.array([100.0] * 4), np.array([100.0] * 4 + [0.0] * 4)

    assert tsb(stopped, 0.1, 0.5) < 0.1 * tsb(buying, 0.1, 0.5)


def test_tsb_ignores_zeros_before_the_first_purchase():
    assert tsb(np.array([0.0, 0.0, 50.0]), 0.1, 0.1) == pytest.approx(50.0)


def test_adida_spreads_each_bucket_back_over_its_cycles():
    # Buys every other cycle: buckets of 2 cycles each hold 60 items.
    assert adida(np.array([60.0, 0.0] * 4), alpha=0.5) == pytest.approx(30.0)


def test_both_strategies_return_zero_for_a_series_that_never_bought():
    assert tsb(np.zeros(4), 0.1, 0.1) == adida(np.zeros(4), 0.1) == 0.0


def test_registered_candidate_answers_every_target_series():
    history, targets = split(panel([10.0, 0, 10, 0, 10, 0, 10, 0], [5.0] * 8), 6)

    predictions = REGISTRY.build(IntermittentParams(strategy="tsb")).fit_predict(history, targets)

    assert set(predictions["serie"]) == {"A", "B"}
    assert (predictions["items_pred"] >= 0).all()
