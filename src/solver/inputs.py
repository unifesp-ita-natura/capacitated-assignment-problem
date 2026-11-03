"""Adapt forecast and generator data into the input dicts the MIP and SA solvers expect."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.generate import generator

ProjectedDemand = dict[tuple[int, int, int], float]  # (sector, day, combination) -> qty


@dataclass(frozen=True)
class ProblemInstance:
    """Block-assignment instance shared by the MIP, matheuristic, and SA solvers.

    `days_by_cycle` holds the days each cycle's balance objective is measured on
    (see `balance_days_by_cycle`); capacity is enforced on every day in `days`.

    `current_assignment` is the dense As-Is map (sector -> combination); the
    MIP's sparse `{(sector, combination): 1}` churn-constraint form is derived
    internally from it where needed, instead of being carried as a second
    representation here.

    `cd_sector_shares` (`{cd: {sector: historical share}}`), when set, lets the
    MIP split a sector's demand across CDs; otherwise `cd_sectors` is a hard
    100% partition.
    """

    sectors: list[int]
    combinations: list[int]
    days: list[int]
    cd_sectors: dict[int, list[int]]
    daily_capacity: dict[tuple[int, int], float]
    projected_demand: ProjectedDemand
    current_assignment: dict[int, int]
    max_churn: float
    days_by_cycle: dict[int, list[int]]
    cd_sector_shares: dict[int, dict[int, float]] | None = None


def build_id_maps(sectors: list[str]) -> tuple[dict[str, int], dict[int, tuple[int, int]]]:
    """Sector-label -> int id map, and combination-id -> (block, sublock) map."""
    sector_ids = {sector: i + 1 for i, sector in enumerate(sectors)}
    combo_slots = {i + 1: slot for i, slot in enumerate(generator.SLOTS)}
    return sector_ids, combo_slots


def build_projected_demand(
    combined_forecast: pd.DataFrame,
    sector_ids: dict[str, int],
    combo_slots: dict[int, tuple[int, int]],
    cycle_span: int,
    first_future_cycle_id: int,
) -> dict[tuple[int, int, int], float]:
    """Every (sector, day-id, combination) forecast quantity, in itens.

    A sector's forecast order-share curve is fixed; only where it *lands* shifts
    with the (block, sublock) combination it is assigned to (each combination
    opens on a different day-in-cycle) and with which future cycle the row
    belongs to. `day_id` extends the day-in-cycle axis by `cycle_offset *
    cycle_span` so each future cycle gets its own non-overlapping block of days
    (`cycle_offset = row.cycle_id - first_future_cycle_id`), letting a single
    shared assignment be checked against every future cycle's demand at once.
    """
    projected_demand: dict[tuple[int, int, int], float] = {}
    for row in combined_forecast.itertuples(index=False):
        sector_id = sector_ids[row.sector]
        cycle_offset = row.cycle_id - first_future_cycle_id
        for combo_id, (block, sublock) in combo_slots.items():
            slot_day = generator.slot_start_day(block, sublock)
            day_id = cycle_offset * cycle_span + slot_day + row.offset
            key = (sector_id, day_id, combo_id)
            projected_demand[key] = projected_demand.get(key, 0.0) + float(row.forecast_items)
    return projected_demand


def group_days_by_cycle(days: list[int], cycle_span: int) -> dict[int, list[int]]:
    """Group extended day ids by the future cycle (0-indexed) they belong to.

    Used to scope each solver's demand-balance objective to a single cycle's own
    days, since capacity is legitimately checked per absolute day across the
    whole horizon but "balance" should not be inflated by cycle-to-cycle trend.
    """
    groups: dict[int, list[int]] = {}
    for day in days:
        groups.setdefault((day - 1) // cycle_span, []).append(day)
    return groups


def balance_days_by_cycle(
    days: list[int], cycle_spacing: int, ramp: int = len(generator.SLOTS) - 1
) -> dict[int, list[int]]:
    """Group the days where every slot is in play by cycle, skipping the horizon's two ramps.

    Cycles open `cycle_spacing` days apart and overlap, so the middle of the horizon
    is already "full". Only the first `ramp` days (later slots have not opened yet)
    and the last `ramp` days (earlier slots have already closed) see a partial
    portfolio; those are left out of the balance objective. Cycles with no days left
    are dropped.
    """
    first, last = min(days), max(days)
    steady = [day for day in days if first + ramp <= day <= last - ramp]
    if not steady:
        raise ValueError(f"no balance days left after trimming {ramp} days at each end")
    return group_days_by_cycle(steady, cycle_spacing)


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


def top_cd_per_sector(shares: dict[str, dict[int, float]]) -> dict[str, int]:
    """Each sector's dominant CD: the one with the largest historical share.

    Ties are broken by lowest CD code, for determinism.
    """
    return {
        sector: min(sector_shares, key=lambda cd: (-sector_shares[cd], cd))
        for sector, sector_shares in shares.items()
    }


def build_cd_sectors(
    sector_to_cd: dict[str, int], sector_ids: dict[str, int]
) -> dict[int, list[int]]:
    """CD code -> list of sector ids assigned to it."""
    cd_sectors: dict[int, list[int]] = {}
    for sector, cd_code in sector_to_cd.items():
        cd_sectors.setdefault(cd_code, []).append(sector_ids[sector])
    return cd_sectors


def effective_cd_sector_shares(instance: ProblemInstance) -> dict[int, dict[int, float]]:
    """The instance's historical shares, or a hard 100% split from `cd_sectors` when absent."""
    if instance.cd_sector_shares is not None:
        return {cd: dict(shares) for cd, shares in instance.cd_sector_shares.items()}
    return {cd: dict.fromkeys(members, 1.0) for cd, members in instance.cd_sectors.items()}


def build_cd_sector_shares(
    shares: dict[str, dict[int, float]], sector_ids: dict[str, int]
) -> dict[int, dict[int, float]]:
    """CD code -> {sector id: historical share}, inverting each sector's {cd: share} map.

    Unlike `build_cd_sectors`'s hard partition, a sector id can appear under more
    than one CD here, weighted by its historical share of that CD's demand.
    """
    cd_sector_shares: dict[int, dict[int, float]] = {}
    for sector, sector_shares in shares.items():
        for cd_code, share in sector_shares.items():
            cd_sector_shares.setdefault(cd_code, {})[sector_ids[sector]] = share
    return cd_sector_shares


def build_current_assignment_sa(
    assignment: dict[str, tuple[int, int]],
    sector_ids: dict[str, int],
    combo_ids: dict[tuple[int, int], int],
) -> dict[int, int]:
    """As-is assignment as the dense {sector_id: combo_id} mapping the SA solver expects."""
    return {sector_ids[sector]: combo_ids[slot] for sector, slot in assignment.items()}
