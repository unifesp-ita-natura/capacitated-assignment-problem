"""Tests for the within-window (shape) forecast strategies."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.shape import (
    shape_last_cycle,
    shape_median,
    shape_plain_average,
    shape_recency_weighted,
    shape_shrinkage,
)


@pytest.fixture
def shape_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sector": ["S1", "S1", "S1", "S1"],
            "cycle_id": [1, 1, 2, 2],
            "offset": [0, 1, 0, 1],
            "order_share": [0.6, 0.4, 0.2, 0.8],
        }
    )


def test_shape_plain_average_is_the_unweighted_mean_per_offset(shape_observations):
    shape = shape_plain_average(shape_observations)

    result = shape.set_index("offset")["order_share"]
    assert result[0] == pytest.approx(0.4)
    assert result[1] == pytest.approx(0.6)


def test_shape_median_is_robust_to_an_outlier_cycle():
    observations = pd.DataFrame(
        {
            "sector": ["S1"] * 6,
            "cycle_id": [1, 1, 2, 2, 3, 3],
            "offset": [0, 1, 0, 1, 0, 1],
            "order_share": [0.1, 0.9, 0.1, 0.9, 0.9, 0.1],
        }
    )

    median_shape = shape_median(observations)
    average_shape = shape_plain_average(observations)

    median_offset0 = median_shape.set_index("offset")["order_share"][0]
    average_offset0 = average_shape.set_index("offset")["order_share"][0]
    assert median_offset0 == pytest.approx(0.1)
    assert average_offset0 == pytest.approx((0.1 + 0.1 + 0.9) / 3)


def test_shape_recency_weighted_favors_the_most_recent_cycle():
    observations = pd.DataFrame(
        {
            "sector": ["S1", "S1", "S1", "S1"],
            "cycle_id": [1, 1, 2, 2],
            "offset": [0, 1, 0, 1],
            "order_share": [0.0, 1.0, 1.0, 0.0],
        }
    )

    weighted = shape_recency_weighted(observations, half_life_cycles=1.0)
    plain = shape_plain_average(observations)

    weighted_value = weighted.set_index("offset")["order_share"][0]
    plain_value = plain.set_index("offset")["order_share"][0]
    assert weighted_value > plain_value


def test_shape_last_cycle_uses_only_the_most_recent_cycle(shape_observations):
    shape = shape_last_cycle(shape_observations)

    result = shape.set_index("offset")["order_share"]
    assert result[0] == pytest.approx(0.2)
    assert result[1] == pytest.approx(0.8)


def test_shape_shrinkage_with_zero_shrinkage_matches_plain_average():
    plain_average = pd.DataFrame(
        {
            "sector": ["S1", "S1", "S2", "S2"],
            "offset": [0, 1, 0, 1],
            "order_share": [0.4, 0.6, 0.8, 0.2],
        }
    )

    shrunk = shape_shrinkage(plain_average, shrinkage=0.0)

    pd.testing.assert_series_equal(
        shrunk["order_share"].reset_index(drop=True),
        plain_average["order_share"].reset_index(drop=True),
    )


def test_shape_shrinkage_with_full_shrinkage_matches_the_global_curve():
    plain_average = pd.DataFrame(
        {
            "sector": ["S1", "S1", "S2", "S2"],
            "offset": [0, 1, 0, 1],
            "order_share": [0.4, 0.6, 0.8, 0.2],
        }
    )

    shrunk = shape_shrinkage(plain_average, shrinkage=1.0)

    s1 = shrunk[shrunk["sector"].eq("S1")].set_index("offset")["order_share"]
    s2 = shrunk[shrunk["sector"].eq("S2")].set_index("offset")["order_share"]
    assert s1[0] == pytest.approx(s2[0])
    assert s1[1] == pytest.approx(s2[1])
