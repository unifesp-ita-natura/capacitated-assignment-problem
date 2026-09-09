"""Tests for the shared data-preparation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.data import (
    build_campanha_totals,
    build_shape_observations,
    normalize_shape,
    to_calendar,
)


def test_build_campanha_totals_sums_orders_per_sector_and_campanha():
    orders = pd.DataFrame(
        {
            "campanha_id": [1, 1, 1, 2],
            "sector": ["S1", "S1", "S2", "S1"],
            "orders": [3, 4, 5, 6],
        }
    )

    totals = build_campanha_totals(orders)

    expected = {(1, "S1"): 7, (1, "S2"): 5, (2, "S1"): 6}
    actual = {(row.campanha_id, row.sector): row.campanha_total for row in totals.itertuples()}
    assert actual == expected


def test_build_shape_observations_computes_order_share():
    orders = pd.DataFrame(
        {
            "campanha_id": [1, 1],
            "sector": ["S1", "S1"],
            "offset": [0, 1],
            "orders": [3, 7],
        }
    )
    campanha_totals = pd.DataFrame({"campanha_id": [1], "sector": ["S1"], "campanha_total": [10]})

    observations = build_shape_observations(orders, campanha_totals)

    assert observations["order_share"].tolist() == pytest.approx([0.3, 0.7])


def test_normalize_shape_rescales_shares_to_sum_to_one():
    shape = pd.DataFrame({"sector": ["S1", "S1"], "offset": [0, 1], "order_share": [0.2, 0.2]})

    normalized = normalize_shape(shape)

    assert normalized["order_share"].tolist() == pytest.approx([0.5, 0.5])
    assert normalized.groupby("sector")["order_share"].sum().iloc[0] == pytest.approx(1.0)


def test_to_calendar_maps_offsets_onto_slot_start_days():
    day_offsets = pd.DataFrame({"sector": ["S1", "S1", "S2"], "offset": [0, 1, 0]})
    start_day_by_sector = {"S1": 1, "S2": 6}
    campanha_open_date = pd.Timestamp("2026-01-01")

    calendar = to_calendar(day_offsets, campanha_open_date, start_day_by_sector)

    assert calendar["day_in_cycle"].tolist() == [1, 2, 6]
    assert calendar["order_date"].tolist() == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-06"),
    ]
