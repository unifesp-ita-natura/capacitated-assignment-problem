"""Adapt forecast and generator data into the input dicts the MIP and SA solvers expect."""

from __future__ import annotations

from cap.generate import generator


def build_id_maps(sectors):
    sector_ids = {sector: i + 1 for i, sector in enumerate(sectors)}
    combo_slots = {i + 1: slot for i, slot in enumerate(generator.SLOTS)}
    return sector_ids, combo_slots


def build_projected_demand(
    combined_forecast, sector_ids, combo_slots, cycle_span, first_future_cycle_id
):
    projected_demand: dict[tuple[int, int, int], float] = {}
    for row in combined_forecast.itertuples(index=False):
        sector_id = sector_ids[row.sector]
        cycle_offset = row.cycle_id - first_future_cycle_id
        for combo_id, (block, sublock) in combo_slots.items():
            slot_day = generator.slot_start_day(block, sublock)
            day_id = cycle_offset * cycle_span + slot_day + row.offset
            projected_demand[(sector_id, day_id, combo_id)] = float(row.forecast_items)
    return projected_demand


def group_days_by_cycle(days, cycle_span):
    groups: dict[int, list[int]] = {}
    for day in days:
        groups.setdefault((day - 1) // cycle_span, []).append(day)
    return groups


def build_daily_capacity(cd_codes, days, capacity_multiplier: float = 1.0):
    return {
        (cd_code, day): capacity_multiplier * generator.CD_CAPACITIES[cd_code]
        for cd_code in cd_codes
        for day in days
    }


def build_cd_sectors(sector_to_cd, sector_ids):
    cd_sectors: dict[int, list[int]] = {}
    for sector, cd_code in sector_to_cd.items():
        cd_sectors.setdefault(cd_code, []).append(sector_ids[sector])
    return cd_sectors


def build_current_assignment_mip(assignment, sector_ids, combo_ids):
    return {(sector_ids[sector], combo_ids[slot]): 1 for sector, slot in assignment.items()}


def build_current_assignment_sa(assignment, sector_ids, combo_ids):
    return {sector_ids[sector]: combo_ids[slot] for sector, slot in assignment.items()}
