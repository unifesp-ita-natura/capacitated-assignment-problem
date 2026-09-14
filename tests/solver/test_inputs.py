"""Tests for the forecast/generator -> solver-input adapter functions."""

from __future__ import annotations

import pandas as pd
import pytest

from src.generate import generator
from src.solver.inputs import (
    build_cd_sectors,
    build_current_assignment_mip,
    build_current_assignment_sa,
    build_daily_capacity,
    build_id_maps,
    build_projected_demand,
)


@pytest.fixture
def sectors() -> list[str]:
    return ["S01", "S02", "S03"]


@pytest.fixture
def combined_forecast() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sector": ["S01", "S01", "S02"],
            "offset": [0, 1, 0],
            "forecast_items": [10.0, 20.0, 30.0],
        }
    )


def test_build_id_maps_assigns_contiguous_1_indexed_sector_ids(sectors):
    sector_ids, _ = build_id_maps(sectors)

    assert sector_ids == {"S01": 1, "S02": 2, "S03": 3}


def test_build_id_maps_combo_slots_match_generator_slots_order(sectors):
    _, combo_slots = build_id_maps(sectors)

    assert combo_slots == {i + 1: slot for i, slot in enumerate(generator.SLOTS)}
    # spot-check a couple of entries against the raw generator SLOTS list
    assert combo_slots[1] == generator.SLOTS[0]
    assert combo_slots[len(generator.SLOTS)] == generator.SLOTS[-1]


def test_build_projected_demand_has_one_entry_per_row_per_combination(sectors, combined_forecast):
    sector_ids, combo_slots = build_id_maps(sectors)

    projected_demand = build_projected_demand(combined_forecast, sector_ids, combo_slots)

    assert len(projected_demand) == len(combined_forecast) * len(combo_slots)


def test_build_projected_demand_day_id_uses_slot_start_day_plus_offset(sectors, combined_forecast):
    sector_ids, combo_slots = build_id_maps(sectors)

    projected_demand = build_projected_demand(combined_forecast, sector_ids, combo_slots)

    for combo_id, (block, sublock) in combo_slots.items():
        expected_day = generator.slot_start_day(block, sublock) + 1  # S01's offset=1 row
        assert projected_demand[(sector_ids["S01"], expected_day, combo_id)] == 20.0


def test_build_daily_capacity_defaults_to_real_cd_capacities():
    cd_codes = [2700, 2800]
    days = [1, 2, 3]

    daily_capacity = build_daily_capacity(cd_codes, days)

    assert daily_capacity == {
        (cd_code, day): generator.CD_CAPACITIES[cd_code] for cd_code in cd_codes for day in days
    }


def test_build_daily_capacity_scales_by_capacity_multiplier():
    cd_codes = [2700, 5300]
    days = [1, 2]
    multiplier = 0.5

    daily_capacity = build_daily_capacity(cd_codes, days, capacity_multiplier=multiplier)

    for cd_code in cd_codes:
        for day in days:
            assert daily_capacity[(cd_code, day)] == multiplier * generator.CD_CAPACITIES[cd_code]


def test_build_cd_sectors_groups_sector_ids_by_cd(sectors):
    sector_ids, _ = build_id_maps(sectors)
    sector_to_cd = {"S01": 2700, "S02": 2700, "S03": 5300}

    cd_sectors = build_cd_sectors(sector_to_cd, sector_ids)

    assert cd_sectors == {2700: [1, 2], 5300: [3]}


def test_build_current_assignment_mip_produces_sparse_indicator_dict(sectors):
    sector_ids, combo_slots = build_id_maps(sectors)
    combo_ids = {slot: combo_id for combo_id, slot in combo_slots.items()}
    assignment = {"S01": generator.SLOTS[0], "S02": generator.SLOTS[3]}

    current_assignment_mip = build_current_assignment_mip(assignment, sector_ids, combo_ids)

    assert current_assignment_mip == {
        (sector_ids["S01"], combo_ids[generator.SLOTS[0]]): 1,
        (sector_ids["S02"], combo_ids[generator.SLOTS[3]]): 1,
    }


def test_build_current_assignment_sa_produces_dense_sector_to_combo_dict(sectors):
    sector_ids, combo_slots = build_id_maps(sectors)
    combo_ids = {slot: combo_id for combo_id, slot in combo_slots.items()}
    assignment = {"S01": generator.SLOTS[0], "S02": generator.SLOTS[3]}

    current_assignment_sa = build_current_assignment_sa(assignment, sector_ids, combo_ids)

    assert current_assignment_sa == {
        sector_ids["S01"]: combo_ids[generator.SLOTS[0]],
        sector_ids["S02"]: combo_ids[generator.SLOTS[3]],
    }
