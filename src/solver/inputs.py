"""Adapt forecast and generator data into the input dicts the MIP and SA solvers expect."""

from __future__ import annotations

import pandas as pd

from src.generate import generator


def build_id_maps(sectors: list[str]) -> tuple[dict[str, int], dict[int, tuple[int, int]]]:
    """Sector-label -> int id map, and combination-id -> (block, sublock) map."""
    sector_ids = {sector: i + 1 for i, sector in enumerate(sectors)}
    combo_slots = {i + 1: slot for i, slot in enumerate(generator.SLOTS)}
    return sector_ids, combo_slots


def build_projected_demand(
    combined_forecast: pd.DataFrame,
    sector_ids: dict[str, int],
    combo_slots: dict[int, tuple[int, int]],
) -> dict[tuple[int, int, int], float]:
    """Every (sector, day-in-cycle, combination) forecast quantity, in itens.

    A sector's forecast order-share curve is fixed; only where it *lands* in the
    cycle shifts with the (block, sublock) combination it is assigned to (each
    combination opens on a different day-in-cycle). This recomputes that landing
    day for every combination so the solvers can compare all of them.
    """
    projected_demand: dict[tuple[int, int, int], float] = {}
    for row in combined_forecast.itertuples(index=False):
        sector_id = sector_ids[row.sector]
        for combo_id, (block, sublock) in combo_slots.items():
            day_id = generator.slot_start_day(block, sublock) + row.offset
            projected_demand[(sector_id, day_id, combo_id)] = float(row.forecast_items)
    return projected_demand


def build_daily_capacity(
    cd_codes: list[int], days: list[int], capacity_multiplier: float = 1.0
) -> dict[tuple[int, int], float]:
    """Flat per-day itens capacity for each CD, from its real daily throughput limit.

    `capacity_multiplier` is an optional headroom/derate factor applied on top of
    the real `CD_CAPACITIES` limit (1.0 leaves it unchanged).
    """
    return {
        (cd_code, day): capacity_multiplier * generator.CD_CAPACITIES[cd_code]
        for cd_code in cd_codes
        for day in days
    }


def build_cd_sectors(
    sector_to_cd: dict[str, int], sector_ids: dict[str, int]
) -> dict[int, list[int]]:
    """CD code -> list of sector ids assigned to it."""
    cd_sectors: dict[int, list[int]] = {}
    for sector, cd_code in sector_to_cd.items():
        cd_sectors.setdefault(cd_code, []).append(sector_ids[sector])
    return cd_sectors


def build_current_assignment_mip(
    assignment: dict[str, tuple[int, int]],
    sector_ids: dict[str, int],
    combo_ids: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    """As-is assignment as the sparse {(sector_id, combo_id): 1} mapping the MIP model expects."""
    return {(sector_ids[sector], combo_ids[slot]): 1 for sector, slot in assignment.items()}


def build_current_assignment_sa(
    assignment: dict[str, tuple[int, int]],
    sector_ids: dict[str, int],
    combo_ids: dict[tuple[int, int], int],
) -> dict[int, int]:
    """As-is assignment as the dense {sector_id: combo_id} mapping the SA solver expects."""
    return {sector_ids[sector]: combo_ids[slot] for sector, slot in assignment.items()}
