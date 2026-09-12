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
    build_cycle_window_shapes,
    build_demand_level,
    build_demand_shape,
    build_sector_metric_ratios,
    build_sector_shape_traits,
    build_sector_volume_parameters,
    expected_orders,
    generate_orders,
    generate_sector_cycle_orders,
    hump_alpha,
    realize_window_shape,
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
def shape_traits(rng) -> dict[str, dict]:
    return build_sector_shape_traits(rng, ["S1", "S2"])


def test_build_sector_shape_traits_covers_every_sector(shape_traits):
    assert set(shape_traits) == {"S1", "S2"}


def test_build_sector_shape_traits_within_default_ranges(shape_traits):
    assert all(0.3 <= trait["peak_frac"] <= 0.7 for trait in shape_traits.values())
    assert all(10 <= trait["concentration"] <= 40 for trait in shape_traits.values())


def test_realize_window_shape_matches_requested_length(rng, shape_traits):
    shape = realize_window_shape(rng, shape_traits["S1"], window_length=10)

    assert len(shape) == 10


def test_realize_window_shape_is_normalized(rng, shape_traits):
    shape = realize_window_shape(rng, shape_traits["S1"], window_length=10)

    assert shape.sum() == pytest.approx(1.0)


def test_build_cycle_window_shapes_matches_requested_lengths(rng, shape_traits):
    shapes = build_cycle_window_shapes(rng, ["S1", "S2"], shape_traits, window_length=7)

    assert {sector: len(shape) for sector, shape in shapes.items()} == {"S1": 7, "S2": 7}


@pytest.fixture
def metric_ratios(rng) -> dict[str, dict[str, float]]:
    return build_sector_metric_ratios(rng, ["S1", "S2"])


def test_build_sector_metric_ratios_covers_every_sector(metric_ratios):
    assert set(metric_ratios) == {"S1", "S2"}


def test_build_sector_metric_ratios_within_default_ranges(metric_ratios):
    assert all(1.0 <= ratios["volumes_per_order"] <= 3.0 for ratios in metric_ratios.values())
    assert all(1.5 <= ratios["itens_per_order"] <= 5.0 for ratios in metric_ratios.values())


def test_expected_orders_scales_linearly_with_each_factor():
    base = expected_orders(
        "S1", window_day_share=0.1, sector_baseline=40, sector_factor=1.0, cycle_factor=1.0
    )
    doubled_baseline = expected_orders(
        "S1", window_day_share=0.1, sector_baseline=80, sector_factor=1.0, cycle_factor=1.0
    )

    assert doubled_baseline == pytest.approx(2 * base)


def test_expected_orders_uses_module_default_window_length():
    with_default = expected_orders("S1", 0.1, 40, 1.0, 1.0)
    with_explicit = expected_orders("S1", 0.1, 40, 1.0, 1.0, window_length=WINDOW_LENGTH)

    assert with_default == with_explicit


@pytest.fixture
def sector_cycle_rows(rng) -> list[dict]:
    assignment = {"S1": (1, 1)}
    window_shape = np.array([0.5, 0.3, 0.2])
    return generate_sector_cycle_orders(
        rng,
        sector="S1",
        cycle_id=1,
        cycle_start=pd.Timestamp("2026-01-05"),  # Monday
        assignment=assignment,
        cycle_factor=1.0,
        window_shape=window_shape,
        baseline=40,
        factor=1.0,
        metric_ratios={"volumes_per_order": 2.0, "itens_per_order": 3.0},
    )


def test_generate_sector_cycle_orders_has_one_row_per_window_day(sector_cycle_rows):
    assert len(sector_cycle_rows) == 3


def test_generate_sector_cycle_orders_uses_the_assigned_slot(sector_cycle_rows):
    slots = {(row["block"], row["sublock"]) for row in sector_cycle_rows}

    assert slots == {(1, 1)}


def test_generate_sector_cycle_orders_offsets_are_sequential(sector_cycle_rows):
    assert [row["offset"] for row in sector_cycle_rows] == [0, 1, 2]


def test_generate_sector_cycle_orders_produces_nonnegative_counts(sector_cycle_rows):
    assert all(row["orders"] >= 0 for row in sector_cycle_rows)
    assert all(row["volumes"] >= 0 for row in sector_cycle_rows)
    assert all(row["itens"] >= 0 for row in sector_cycle_rows)


def test_generate_sector_cycle_orders_window_start_is_a_business_day(rng):
    assignment = {"S1": (3, 5)}  # last slot of the cycle, likely to roll into a weekend
    window_shape = np.ones(10) / 10

    rows = generate_sector_cycle_orders(
        rng,
        sector="S1",
        cycle_id=1,
        cycle_start=pd.Timestamp("2026-01-05"),  # Monday
        assignment=assignment,
        cycle_factor=1.0,
        window_shape=window_shape,
        baseline=40,
        factor=1.0,
        metric_ratios={"volumes_per_order": 2.0, "itens_per_order": 3.0},
    )

    order_dates = pd.DatetimeIndex([row["order_date"] for row in rows])
    assert order_dates[0].dayofweek <= 4  # the window always opens Mon-Fri
    assert order_dates.dayofweek.max() > 4  # but customers can order through the weekend


@pytest.fixture
def orders_df(rng) -> pd.DataFrame:
    sectors = [f"S{i}" for i in range(3)]
    assignment = build_current_assignment(rng, sectors=sectors)
    cycle_starts = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-02")]
    return generate_orders(rng, sectors, cycle_starts, assignment)


def test_generate_orders_row_count_matches_sectors_cycles_and_window(orders_df):
    assert len(orders_df) == 3 * 2 * WINDOW_LENGTH


def test_generate_orders_has_expected_columns(orders_df):
    assert set(orders_df.columns) == {
        "order_date",
        "cycle_id",
        "day_in_cycle",
        "offset",
        "block",
        "sublock",
        "sector",
        "orders",
        "volumes",
        "itens",
    }


def test_generate_orders_respects_per_cycle_window_lengths(rng):
    sectors = ["S1", "S2"]
    assignment = build_current_assignment(rng, sectors=sectors)
    cycle_starts = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-02")]

    df = generate_orders(
        rng,
        sectors,
        cycle_starts,
        assignment,
        window_lengths={1: 10, 2: 15},
    )

    counts = df.groupby("cycle_id").size()
    assert counts[1] == len(sectors) * 10
    assert counts[2] == len(sectors) * 15


def test_generate_orders_covers_every_cycle(orders_df):
    assert set(orders_df["cycle_id"]) == {1, 2}


def test_generate_orders_covers_every_sector(orders_df):
    assert set(orders_df["sector"]) == {f"S{i}" for i in range(3)}


def test_generate_orders_produces_nonnegative_counts(orders_df):
    assert (orders_df["orders"] >= 0).all()


def test_generate_orders_uses_supplied_cycle_factors():
    sectors = ["S1"]
    assignment = {"S1": (1, 1)}
    cycle_starts = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-02")]

    df = generate_orders(
        np.random.default_rng(1),
        sectors,
        cycle_starts,
        assignment,
        cycle_factors={1: 1.0, 2: 5.0},
    )

    mean_by_cycle = df.groupby("cycle_id")["orders"].mean()
    assert mean_by_cycle[2] > mean_by_cycle[1]


def test_build_demand_level_matches_processed_schema(orders_df):
    level = build_demand_level(orders_df)

    assert list(level.columns) == [
        "cd_setor",
        "ciclo",
        "date",
        "cycle_duration",
        "total_pedidos",
        "total_volumes",
        "total_itens",
    ]


def test_build_demand_level_one_row_per_sector_cycle(orders_df):
    level = build_demand_level(orders_df)

    assert len(level) == orders_df[["sector", "cycle_id"]].drop_duplicates().shape[0]


def test_build_demand_level_cycle_duration_matches_window_length(orders_df):
    level = build_demand_level(orders_df)

    assert (level["cycle_duration"] == WINDOW_LENGTH).all()


def test_build_demand_level_totals_match_raw_sums(orders_df):
    level = build_demand_level(orders_df)
    expected_total = orders_df.loc[
        orders_df["sector"].eq("S0") & orders_df["cycle_id"].eq(1), "orders"
    ].sum()
    actual_total = level.loc[
        level["cd_setor"].eq("S0") & level["ciclo"].eq("1"), "total_pedidos"
    ].iloc[0]

    assert actual_total == expected_total


def test_build_demand_shape_matches_processed_schema(orders_df):
    shape = build_demand_shape(orders_df)

    assert list(shape.columns) == [
        "cd_setor",
        "ciclo",
        "data_pedido",
        "relative_date",
        "total_pedidos",
        "total_volumes",
        "total_itens",
        "ciclo_total_pedidos",
        "ciclo_total_volumes",
        "ciclo_total_itens",
        "share_pedidos",
        "share_volumes",
        "share_itens",
    ]


def test_build_demand_shape_one_row_per_orders_row(orders_df):
    shape = build_demand_shape(orders_df)

    assert len(shape) == len(orders_df)


def test_build_demand_shape_relative_date_within_unit_interval(orders_df):
    shape = build_demand_shape(orders_df)

    assert shape["relative_date"].between(0, 1, inclusive="left").all()


def test_build_demand_shape_shares_sum_to_one_per_sector_cycle(orders_df):
    shape = build_demand_shape(orders_df)
    # only groups with nonzero cycle totals have well-defined shares
    non_degenerate = shape[shape["ciclo_total_pedidos"] > 0]

    totals = non_degenerate.groupby(["cd_setor", "ciclo"])["share_pedidos"].sum()
    assert all(total == pytest.approx(1.0) for total in totals)
