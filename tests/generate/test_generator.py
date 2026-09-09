"""Tests for the AS-IS synthetic order generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.generate.generator import (
    BLOCKS,
    SUBLOCKS,
    WINDOW_LENGTH,
    build_current_assignment,
    build_sector_volume_parameters,
    build_sector_window_shapes,
    expected_orders,
    generate_orders,
    generate_sector_campanha_orders,
    hump_alpha,
    slot_start_day,
    uniform_block_distribution,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_uniform_block_distribution_sums_to_one_and_is_equal():
    dist = uniform_block_distribution([1, 2, 3, 4])

    assert dist == {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}


@pytest.mark.parametrize(
    "block, sublock, expected",
    [
        (1, 1, 1),
        (1, 5, 5),
        (2, 1, 6),
        (3, 5, 15),
    ],
)
def test_slot_start_day(block, sublock, expected):
    assert slot_start_day(block, sublock) == expected


@pytest.fixture
def current_assignment(rng) -> dict[str, tuple[int, int]]:
    sectors = [f"S{i}" for i in range(20)]
    return build_current_assignment(rng, sectors=sectors)


def test_build_current_assignment_covers_every_sector(current_assignment):
    assert set(current_assignment) == {f"S{i}" for i in range(20)}


def test_build_current_assignment_uses_valid_blocks(current_assignment):
    assert {block for block, _ in current_assignment.values()} <= set(BLOCKS)


def test_build_current_assignment_uses_valid_sublocks(current_assignment):
    assert {sublock for _, sublock in current_assignment.values()} <= set(SUBLOCKS)


def test_build_current_assignment_respects_block_weights(rng):
    sectors = [f"S{i}" for i in range(5000)]
    block_weights = {1: 1.0, 2: 0.0, 3: 0.0}

    assignment = build_current_assignment(rng, sectors=sectors, block_weights=block_weights)

    assert {block for block, _ in assignment.values()} == {1}


def test_build_current_assignment_is_deterministic_for_a_given_rng_state():
    sectors = ["S1", "S2", "S3"]

    first = build_current_assignment(np.random.default_rng(0), sectors=sectors)
    second = build_current_assignment(np.random.default_rng(0), sectors=sectors)

    assert first == second


@pytest.fixture
def volume_parameters(rng) -> tuple[dict[str, float], dict[str, float]]:
    sectors = [f"S{i}" for i in range(50)]
    return build_sector_volume_parameters(
        rng, sectors, min_volume=20, max_volume=60, variance_percentage=0.15
    )


def test_build_sector_volume_parameters_covers_every_sector(volume_parameters):
    baseline, factor = volume_parameters

    assert set(baseline) == set(factor) == {f"S{i}" for i in range(50)}


def test_build_sector_volume_parameters_baseline_within_bounds(volume_parameters):
    baseline, _ = volume_parameters

    assert all(20 <= value <= 60 for value in baseline.values())


def test_build_sector_volume_parameters_factor_within_bounds(volume_parameters):
    _, factor = volume_parameters

    assert all(0.85 <= value <= 1.15 for value in factor.values())


def test_hump_alpha_sums_to_concentration():
    alpha = hump_alpha(window_length=21, concentration=20, peak_frac=0.5)

    assert alpha.sum() == pytest.approx(20, rel=1e-6)


def test_hump_alpha_peaks_near_peak_frac():
    alpha = hump_alpha(window_length=21, concentration=20, peak_frac=0.5)

    assert np.argmax(alpha) == round(0.5 * 20)


@pytest.fixture
def window_shapes(rng) -> dict[str, np.ndarray]:
    sector_meta = {"S1": {"window_length": 10}, "S2": {"window_length": 5}}
    return build_sector_window_shapes(rng, sector_meta)


def test_build_sector_window_shapes_matches_requested_lengths(window_shapes):
    assert {sector: len(shape) for sector, shape in window_shapes.items()} == {"S1": 10, "S2": 5}


def test_build_sector_window_shapes_are_normalized(window_shapes):
    assert all(shape.sum() == pytest.approx(1.0) for shape in window_shapes.values())


def test_expected_orders_scales_linearly_with_each_factor():
    base = expected_orders(
        "S1", window_day_share=0.1, sector_baseline=40, sector_factor=1.0, campanha_factor=1.0
    )
    doubled_baseline = expected_orders(
        "S1", window_day_share=0.1, sector_baseline=80, sector_factor=1.0, campanha_factor=1.0
    )

    assert doubled_baseline == pytest.approx(2 * base)


def test_expected_orders_uses_module_default_window_length():
    with_default = expected_orders("S1", 0.1, 40, 1.0, 1.0)
    with_explicit = expected_orders("S1", 0.1, 40, 1.0, 1.0, window_length=WINDOW_LENGTH)

    assert with_default == with_explicit


@pytest.fixture
def sector_campanha_rows(rng) -> list[dict]:
    assignment = {"S1": (1, 1)}
    window_shape = np.array([0.5, 0.3, 0.2])
    return generate_sector_campanha_orders(
        rng,
        sector="S1",
        campanha_id=1,
        campanha_start=pd.Timestamp("2026-01-05"),  # Monday
        assignment=assignment,
        campanha_factor=1.0,
        window_shape=window_shape,
        baseline=40,
        factor=1.0,
    )


def test_generate_sector_campanha_orders_has_one_row_per_window_day(sector_campanha_rows):
    assert len(sector_campanha_rows) == 3


def test_generate_sector_campanha_orders_uses_the_assigned_slot(sector_campanha_rows):
    slots = {(row["block"], row["sublock"]) for row in sector_campanha_rows}

    assert slots == {(1, 1)}


def test_generate_sector_campanha_orders_offsets_are_sequential(sector_campanha_rows):
    assert [row["offset"] for row in sector_campanha_rows] == [0, 1, 2]


def test_generate_sector_campanha_orders_produces_nonnegative_counts(sector_campanha_rows):
    assert all(row["orders"] >= 0 for row in sector_campanha_rows)


def test_generate_sector_campanha_orders_window_start_is_a_business_day(rng):
    assignment = {"S1": (3, 5)}  # last slot of the cycle, likely to roll into a weekend
    window_shape = np.ones(10) / 10

    rows = generate_sector_campanha_orders(
        rng,
        sector="S1",
        campanha_id=1,
        campanha_start=pd.Timestamp("2026-01-05"),  # Monday
        assignment=assignment,
        campanha_factor=1.0,
        window_shape=window_shape,
        baseline=40,
        factor=1.0,
    )

    order_dates = pd.DatetimeIndex([row["order_date"] for row in rows])
    assert order_dates[0].dayofweek <= 4  # the window always opens Mon-Fri
    assert order_dates.dayofweek.max() > 4  # but customers can order through the weekend


@pytest.fixture
def orders_df(rng) -> pd.DataFrame:
    sectors = [f"S{i}" for i in range(3)]
    assignment = build_current_assignment(rng, sectors=sectors)
    campanha_starts = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-02")]
    return generate_orders(rng, sectors, campanha_starts, assignment)


def test_generate_orders_row_count_matches_sectors_campanhas_and_window(orders_df):
    assert len(orders_df) == 3 * 2 * WINDOW_LENGTH


def test_generate_orders_has_expected_columns(orders_df):
    assert set(orders_df.columns) == {
        "order_date",
        "campanha_id",
        "day_in_cycle",
        "offset",
        "block",
        "sublock",
        "sector",
        "orders",
    }


def test_generate_orders_covers_every_campanha(orders_df):
    assert set(orders_df["campanha_id"]) == {1, 2}


def test_generate_orders_covers_every_sector(orders_df):
    assert set(orders_df["sector"]) == {f"S{i}" for i in range(3)}


def test_generate_orders_produces_nonnegative_counts(orders_df):
    assert (orders_df["orders"] >= 0).all()


def test_generate_orders_uses_supplied_campanha_factors():
    sectors = ["S1"]
    assignment = {"S1": (1, 1)}
    campanha_starts = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-02")]

    df = generate_orders(
        np.random.default_rng(1),
        sectors,
        campanha_starts,
        assignment,
        campanha_factors={1: 1.0, 2: 5.0},
    )

    mean_by_campanha = df.groupby("campanha_id")["orders"].mean()
    assert mean_by_campanha[2] > mean_by_campanha[1]
