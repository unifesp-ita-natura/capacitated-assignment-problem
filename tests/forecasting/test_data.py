"""Tests for the shared data-preparation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.data import (
    build_cycle_totals,
    build_shape_observations,
    normalize_shape,
    to_calendar,
    validate_demand_metric,
)


def test_validate_demand_metric_accepts_known_metrics():
    assert validate_demand_metric("volumes") == "volumes"


def test_validate_demand_metric_rejects_unknown_metric():
    with pytest.raises(ValueError):
        validate_demand_metric("dinheiro")


def test_build_cycle_totals_sums_the_chosen_metric_per_sector_and_cycle():
    demand_level = pd.DataFrame(
        {
            "cd_setor": ["S1", "S1", "S2"],
            "ciclo": ["1", "2", "1"],
            "total_pedidos": [7, 6, 5],
            "total_volumes": [70, 60, 50],
            "total_itens": [700, 600, 500],
        }
    )

    totals = build_cycle_totals(demand_level, metric="pedidos")

    expected = {(1, "S1"): 7, (2, "S1"): 6, (1, "S2"): 5}
    actual = {(row.cycle_id, row.sector): row.cycle_total for row in totals.itertuples()}
    assert actual == expected


def test_build_cycle_totals_respects_the_chosen_metric():
    demand_level = pd.DataFrame(
        {
            "cd_setor": ["S1"],
            "ciclo": ["1"],
            "total_pedidos": [7],
            "total_volumes": [70],
            "total_itens": [700],
        }
    )

    totals = build_cycle_totals(demand_level, metric="itens")

    assert totals["cycle_total"].tolist() == [700]


def test_build_shape_observations_computes_order_share_and_offset():
    demand_shape = pd.DataFrame(
        {
            "cd_setor": ["S1", "S1"],
            "ciclo": ["1", "1"],
            "relative_date": [0.0, 0.5],
            "share_pedidos": [0.3, 0.7],
            "share_volumes": [0.4, 0.6],
        }
    )
    demand_level = pd.DataFrame({"cd_setor": ["S1"], "ciclo": ["1"], "cycle_duration": [2]})

    observations = build_shape_observations(demand_shape, demand_level, metric="pedidos")

    assert observations["order_share"].tolist() == pytest.approx([0.3, 0.7])
    assert observations["offset"].tolist() == [0, 1]
    assert observations["cycle_id"].tolist() == [1, 1]


def test_build_shape_observations_respects_the_chosen_metric():
    demand_shape = pd.DataFrame(
        {
            "cd_setor": ["S1"],
            "ciclo": ["1"],
            "relative_date": [0.0],
            "share_pedidos": [0.3],
            "share_volumes": [0.4],
        }
    )
    demand_level = pd.DataFrame({"cd_setor": ["S1"], "ciclo": ["1"], "cycle_duration": [1]})

    observations = build_shape_observations(demand_shape, demand_level, metric="volumes")

    assert observations["order_share"].tolist() == pytest.approx([0.4])


def test_normalize_shape_rescales_shares_to_sum_to_one():
    shape = pd.DataFrame({"sector": ["S1", "S1"], "offset": [0, 1], "order_share": [0.2, 0.2]})

    normalized = normalize_shape(shape)

    assert normalized["order_share"].tolist() == pytest.approx([0.5, 0.5])
    assert normalized.groupby("sector")["order_share"].sum().iloc[0] == pytest.approx(1.0)


def test_to_calendar_maps_offsets_onto_slot_start_days():
    day_offsets = pd.DataFrame({"sector": ["S1", "S1", "S2"], "offset": [0, 1, 0]})
    start_day_by_sector = {"S1": 1, "S2": 6}
    cycle_open_date = pd.Timestamp("2026-01-01")  # Thursday

    calendar = to_calendar(day_offsets, cycle_open_date, start_day_by_sector)

    assert calendar["day_in_cycle"].tolist() == [1, 2, 6]
    assert calendar["order_date"].tolist() == [
        pd.Timestamp("2026-01-01"),  # S1's window opens on the cycle's own start (a Thursday)
        pd.Timestamp("2026-01-02"),  # offset 1 is a calendar day later (Friday)
        pd.Timestamp("2026-01-08"),  # S2's window opens on the 6th business day (Thursday)
    ]


def test_to_calendar_lets_orders_land_on_a_weekend_within_a_long_window():
    day_offsets = pd.DataFrame({"sector": ["S1"] * 4, "offset": [0, 1, 2, 3]})
    start_day_by_sector = {"S1": 1}
    cycle_open_date = pd.Timestamp("2026-01-01")  # Thursday

    calendar = to_calendar(day_offsets, cycle_open_date, start_day_by_sector)

    assert calendar["order_date"].tolist() == [
        pd.Timestamp("2026-01-01"),  # Thursday
        pd.Timestamp("2026-01-02"),  # Friday
        pd.Timestamp("2026-01-03"),  # Saturday
        pd.Timestamp("2026-01-04"),  # Sunday
    ]
