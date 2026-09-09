"""Expand parameterized shape strategies across a grid of values for tuning and selection."""

from __future__ import annotations

from collections.abc import Iterable

from src.forecasting import shape
from src.forecasting.shape import SHAPE_STRATEGIES, ShapeStrategy


def make_shape_recency_weighted(half_life_campanhas: float) -> ShapeStrategy:
    """Bind a half-life to `shape_recency_weighted`, producing a plain ShapeStrategy."""
    return lambda observations: shape.shape_recency_weighted(observations, half_life_campanhas)


def make_shape_shrinkage(shrinkage: float) -> ShapeStrategy:
    """Bind a shrinkage factor to `shape_shrinkage`, producing a plain ShapeStrategy."""

    def strategy(observations):
        return shape.shape_shrinkage(shape.shape_plain_average(observations), shrinkage)

    return strategy


def sweep_shape_recency_weighted(
    half_life_campanhas_grid: Iterable[float],
) -> dict[str, ShapeStrategy]:
    """One named recency-weighted strategy per half-life in the grid."""
    return {
        f"recency_weighted_hl={hl:g}": make_shape_recency_weighted(hl)
        for hl in half_life_campanhas_grid
    }


def sweep_shape_shrinkage(shrinkage_grid: Iterable[float]) -> dict[str, ShapeStrategy]:
    """One named shrinkage strategy per shrinkage factor in the grid."""
    return {f"shrinkage_s={s:g}": make_shape_shrinkage(s) for s in shrinkage_grid}


def build_shape_strategy_registry(
    half_life_campanhas_grid: Iterable[float] = (2.0,),
    shrinkage_grid: Iterable[float] = (0.3,),
) -> dict[str, ShapeStrategy]:
    """Every shape strategy, with the parameterized ones expanded across their grids."""
    return {
        **SHAPE_STRATEGIES,
        **sweep_shape_recency_weighted(half_life_campanhas_grid),
        **sweep_shape_shrinkage(shrinkage_grid),
    }
