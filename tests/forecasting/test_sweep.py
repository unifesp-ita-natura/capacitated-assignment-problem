"""Tests for expanding parameterized shape strategies across a grid of values."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.sweep import (
    build_shape_strategy_registry,
    make_shape_recency_weighted,
    make_shape_shrinkage,
    sweep_shape_recency_weighted,
    sweep_shape_shrinkage,
)


@pytest.fixture
def shape_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sector": ["S1", "S1", "S1", "S1"],
            "campanha_id": [1, 1, 2, 2],
            "offset": [0, 1, 0, 1],
            "order_share": [0.6, 0.4, 0.2, 0.8],
        }
    )


def test_make_shape_recency_weighted_binds_the_half_life(shape_observations):
    from src.forecasting.shape import shape_recency_weighted

    strategy = make_shape_recency_weighted(1.0)

    assert strategy(shape_observations).equals(
        shape_recency_weighted(shape_observations, half_life_campanhas=1.0)
    )


def test_make_shape_shrinkage_binds_the_shrinkage_factor(shape_observations):
    from src.forecasting.shape import shape_plain_average, shape_shrinkage

    strategy = make_shape_shrinkage(0.3)
    plain_average = shape_plain_average(shape_observations)

    assert strategy(shape_observations).equals(shape_shrinkage(plain_average, shrinkage=0.3))


def test_sweep_shape_recency_weighted_names_one_variant_per_half_life():
    variants = sweep_shape_recency_weighted([1.0, 2.0, 4.0])

    assert set(variants) == {
        "recency_weighted_hl=1",
        "recency_weighted_hl=2",
        "recency_weighted_hl=4",
    }


def test_sweep_shape_shrinkage_names_one_variant_per_shrinkage_value():
    variants = sweep_shape_shrinkage([0.1, 0.3, 0.5])

    assert set(variants) == {"shrinkage_s=0.1", "shrinkage_s=0.3", "shrinkage_s=0.5"}


def test_build_shape_strategy_registry_returns_every_named_strategy():
    registry = build_shape_strategy_registry(half_life_campanhas_grid=[2.0], shrinkage_grid=[0.3])

    assert set(registry) == {
        "plain_average",
        "median",
        "last_campanha",
        "recency_weighted_hl=2",
        "shrinkage_s=0.3",
    }


def test_build_shape_strategy_registry_expands_a_multi_value_grid():
    registry = build_shape_strategy_registry(
        half_life_campanhas_grid=[1.0, 2.0], shrinkage_grid=[0.1, 0.5]
    )

    assert "recency_weighted_hl=1" in registry
    assert "recency_weighted_hl=2" in registry
    assert "shrinkage_s=0.1" in registry
    assert "shrinkage_s=0.5" in registry
