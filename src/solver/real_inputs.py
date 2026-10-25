"""Build a `ProblemInstance` from the real-data daily forecast and the observed As-Is calendar."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.forecasting.dataset import cycle_calendar
from src.generate import generator
from src.solver import inputs as solver_inputs
from src.solver.inputs import ProblemInstance

AS_IS_QUANTILE = 0.05  # robust "cycle start": a few sectors open far earlier than the rest


class RealInstance(NamedTuple):
    """An instance plus the label maps needed to read its solution back in business terms."""

    instance: ProblemInstance
    sector_ids: dict[str, int]
    sector_to_cd: dict[str, int]


def sector_curves(daily: pd.DataFrame, cycles: list[str]) -> pd.DataFrame:
    """Items per (sector, cycle index, 0-based day offset), summed over the sector's cities."""
    frame = daily.dropna(subset=["items_pred"]).copy()
    frame["cycle_id"] = frame["CICLOS"].map({cycle: i for i, cycle in enumerate(cycles)})
    frame["offset"] = frame["relative_day"] - 1
    curves = frame.groupby(["cd_setor", "cycle_id", "offset"], as_index=False)["items_pred"].sum()
    return curves.rename(columns={"cd_setor": "sector", "items_pred": "forecast_items"})


def cycle_spacing(raw: pd.DataFrame, cycles: list[str]) -> int:
    """Median days between consecutive cycle openings (of `cycles`, else of the whole base)."""
    calendar = cycle_calendar(raw).sort_values("opening_date")
    chosen = calendar[calendar["CICLOS"].isin(cycles)] if len(cycles) > 1 else calendar
    return int(chosen["opening_date"].diff().dt.days.median())


def dominant_cd(raw: pd.DataFrame, sectors: list[str]) -> dict[str, int]:
    """The CD that receives most of each sector's items."""
    selected = raw[raw["cd_setor"].isin(sectors)]
    items = selected.groupby(["cd_setor", "cd_cd"])["total_itens_mascarado"].sum()
    top = items.groupby(level="cd_setor").idxmax()
    return {sector: int(cd) for sector, (_, cd) in top.items()}


def infer_as_is(raw: pd.DataFrame, cycle: str, sectors: list[str]) -> dict[str, int]:
    """Each sector's slot day (= combination id) from when it opened relative to its peers.

    The base has no block/sub-block columns, so a sector's slot is its opening date
    measured from a robust cycle start and clipped to the 15 available slots.
    Sectors absent from `cycle` take the median slot.
    """
    openings = raw[raw["CICLOS"].eq(cycle)].groupby("cd_setor")["Dt Abertura"].min()
    start = openings.sort_values().iloc[int(AS_IS_QUANTILE * (len(openings) - 1))]
    slots = ((openings - start).dt.days + 1).clip(1, len(generator.SLOTS))
    fallback = int(np.median(slots))
    return {sector: int(slots.get(sector, fallback)) for sector in sectors}


def build_real_instance(
    daily: pd.DataFrame,
    raw: pd.DataFrame,
    cycles: list[str],
    as_is_cycle: str,
    max_churn: float,
    capacity_multiplier: float = 1.0,
    max_sectors: int | None = None,
    fixed_sectors: dict[str, int] | None = None,
) -> RealInstance:
    """Assemble the solver instance for the forecast `cycles` (in order) of the real data.

    `fixed_sectors` pins sector labels to combination ids the solvers must keep.
    """
    curves = _largest_sectors(sector_curves(daily, cycles), max_sectors)
    labels = sorted(curves["sector"].unique())
    validate_fixed_sectors(fixed_sectors or {}, labels)
    sector_ids, combo_slots = solver_inputs.build_id_maps(labels)
    spacing = cycle_spacing(raw, cycles)
    demand = solver_inputs.build_projected_demand(curves, sector_ids, combo_slots, spacing, 0)
    days = sorted({day for _, day, _ in demand})
    sector_to_cd = dominant_cd(raw, labels)
    cd_sectors = solver_inputs.build_cd_sectors(sector_to_cd, sector_ids)
    as_is = infer_as_is(raw, as_is_cycle, labels)
    max_moves = max_churn * len(labels)
    fixed = _pinned_ids(fixed_sectors or {}, sector_ids, as_is, max_moves)
    instance = ProblemInstance(
        sectors=list(sector_ids.values()),
        combinations=list(combo_slots),
        days=days,
        cd_sectors=cd_sectors,
        daily_capacity=solver_inputs.build_daily_capacity(
            list(cd_sectors), days, capacity_multiplier
        ),
        projected_demand=demand,
        current_assignment={sector_ids[s]: slot for s, slot in as_is.items()},
        max_churn=max_moves,
        days_by_cycle=solver_inputs.balance_days_by_cycle(days, spacing),
        fixed_assignment=fixed,
    )
    return RealInstance(instance, sector_ids, sector_to_cd)


def _pinned_ids(
    fixed_sectors: dict[str, int],
    sector_ids: dict[str, int],
    as_is: dict[str, int],
    max_moves: float,
) -> dict[int, int]:
    """The pins keyed by sector id, checked against the churn budget."""
    fixed = {sector_ids[s]: combo for s, combo in fixed_sectors.items()}
    validate_fixed_churn(fixed, {sector_ids[s]: c for s, c in as_is.items()}, max_moves)
    return fixed


def validate_fixed_sectors(fixed_sectors: dict[str, int], labels: list[str]) -> None:
    """Raise if a pinned sector is unknown or pinned to a combination that does not exist."""
    unknown = sorted(set(fixed_sectors) - set(labels))
    if unknown:
        raise ValueError(f"fixed_sectors names unknown sectors: {unknown}")
    bad = {s: c for s, c in fixed_sectors.items() if not 1 <= c <= len(generator.SLOTS)}
    if bad:
        raise ValueError(f"fixed_sectors combination outside 1..{len(generator.SLOTS)}: {bad}")


def validate_fixed_churn(fixed: dict[int, int], as_is: dict[int, int], max_churn: float) -> None:
    """Raise if the pins alone already move more sectors than the churn budget allows."""
    moved = sum(1 for sector, combo in fixed.items() if as_is[sector] != combo)
    if moved > max_churn:
        raise ValueError(f"fixed_sectors move {moved} sectors, above the churn budget {max_churn}")


def _largest_sectors(curves: pd.DataFrame, max_sectors: int | None) -> pd.DataFrame:
    """Restrict to the `max_sectors` sectors with the most forecast items (all if None)."""
    if max_sectors is None:
        return curves
    totals = curves.groupby("sector")["forecast_items"].sum().nlargest(max_sectors)
    return curves[curves["sector"].isin(totals.index)]
