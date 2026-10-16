"""Solve the block-assignment model across a sector/cycle size sweep with each
configured solver, and write a comparison table.

Capacity is kept deliberately loose (far above any possible demand) so the
sweep isolates how solve time/behavior scales with instance size, rather than
with capacity feasibility.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Mapping
from pathlib import Path

from src.persistence import SolveResult, configure_tracking, log_solve_result
from src.solver.mip.block_assignment import ProjectedDemand, run_block_assignment_mip

# Add/remove Pyomo-registered solver names to compare different backends
# against the same model. To compare different *formulations* instead (or in
# addition), swap the generated instance below for a list of
# instances/builders and loop over both.
SOLVERS_TO_COMPARE = ["gurobi", "appsi_highs"]

OUTPUT_PATH = "experiments/compare_modeling/outputs/results.csv"

# How many sectors the synthetic instance has, swept from a small toy size up
# to a large one. Step of 200 with 2000 appended so the range always lands
# exactly on both endpoints.
SECTOR_SWEEP = list(range(40, 2000, 200)) + [2000]

# How many forecast cycles the horizon spans, crossed with SECTOR_SWEEP.
CYCLE_SWEEP = [1, 2, 4, 8]

# Each cycle's length (in days) is sampled independently (with replacement)
# from this range, so cycles aren't all the same size.
CYCLE_LENGTH_RANGE = (19, 23)

# Combinations = the sector's choice of which of the first N_COMBINATIONS
# days of a cycle it opens on (1st open day, 2nd open day, ...). Fixed
# regardless of cycle length; must not exceed CYCLE_LENGTH_RANGE's minimum,
# so every combination maps to a real day in every cycle.
N_COMBINATIONS = 15
N_CDS = 2
SEED = 42


def _sample_cycle_lengths(n_cycles: int, rng: random.Random) -> list[int]:
    low, high = CYCLE_LENGTH_RANGE
    return [rng.randint(low, high) for _ in range(n_cycles)]


def _build_cycle_structure(cycle_lengths: list[int]) -> tuple[list[int], dict[int, list[int]]]:
    """Lays `cycle_lengths` end to end into a single day-numbered timeline.

    `combinations` must span every day-of-cycle index this produces, i.e.
    `range(1, max(cycle_lengths) + 1)` — see `generate_instance`.
    """
    days = list(range(1, sum(cycle_lengths) + 1))
    days_by_cycle = {}
    start = 0
    for h, length in enumerate(cycle_lengths):
        days_by_cycle[h] = days[start : start + length]
        start += length
    return days, days_by_cycle


def _assign_sectors_to_cds(sectors: list[int], n_cds: int) -> dict[int, list[int]]:
    cd_sectors: dict[int, list[int]] = {cd: [] for cd in range(1, n_cds + 1)}
    for s in sectors:
        cd_sectors[(s - 1) % n_cds + 1].append(s)
    return cd_sectors


def _peak_combination_by_day(days_by_cycle: Mapping[int, list[int]]) -> dict[int, int]:
    """Day-of-cycle `d` (1-indexed) is the day each repeating cycle peaks on combination `d`."""
    return {
        day: offset + 1
        for cycle_days in days_by_cycle.values()
        for offset, day in enumerate(cycle_days)
    }


def _demand_by_combination(
    demand: float, peak_combination: int, combinations: list[int]
) -> dict[int, float]:
    """A sector's demand on one day: full demand under the day's peak combination, 0 otherwise."""
    return {
        combination: demand if combination == peak_combination else 0.0
        for combination in combinations
    }


def _build_projected_demand(
    sector_demand: Mapping[int, float],
    combinations: list[int],
    days_by_cycle: Mapping[int, list[int]],
) -> ProjectedDemand:
    """Each sector's demand lands fully on day-of-cycle `d` under combination `d`, 0 elsewhere.

    The pattern repeats across cycles — the generalization of the toy
    instance's two-combination, two-day shape.
    """
    peak_combination_by_day = _peak_combination_by_day(days_by_cycle)
    return {
        (s, day, combination): value
        for s, demand in sector_demand.items()
        for day, peak_combination in peak_combination_by_day.items()
        for combination, value in _demand_by_combination(
            demand, peak_combination, combinations
        ).items()
    }


def _build_loose_daily_capacity(
    sector_demand: Mapping[int, float],
    cd_sectors: Mapping[int, list[int]],
    days: list[int],
) -> dict[tuple[int, int], float]:
    """Capacity far above total demand, so it never binds anywhere in the sweep."""
    loose_capacity = sum(sector_demand.values()) * 10
    return {(cd, a): loose_capacity for cd in cd_sectors for a in days}


def generate_instance(
    n_sectors: int,
    n_cycles: int,
    n_cds: int = N_CDS,
    seed: int = SEED,
) -> dict:
    """A synthetic block-assignment instance sized by `n_sectors`/`n_cycles`.

    Each cycle's length is independently sampled from `CYCLE_LENGTH_RANGE`
    (with replacement), so cycles vary in size rather than all being equal.
    `combinations` stays fixed at `N_COMBINATIONS` regardless of cycle
    length (see its module-level comment). Each sector gets one random,
    fixed demand magnitude (see `_build_projected_demand` for how
    combinations distribute it across days). Daily capacity is set to a
    large multiple of total demand so it never binds, keeping the sweep
    focused on assignment/balancing cost rather than feasibility.
    """
    rng = random.Random(seed)
    sectors = list(range(1, n_sectors + 1))
    cycle_lengths = _sample_cycle_lengths(n_cycles, rng)
    combinations = list(range(1, N_COMBINATIONS + 1))
    days, days_by_cycle = _build_cycle_structure(cycle_lengths)
    cd_sectors = _assign_sectors_to_cds(sectors, n_cds)
    sector_demand = {s: rng.uniform(5.0, 50.0) for s in sectors}

    return dict(
        sectors=sectors,
        combinations=combinations,
        days=days,
        cd_sectors=cd_sectors,
        daily_capacity=_build_loose_daily_capacity(sector_demand, cd_sectors, days),
        projected_demand=_build_projected_demand(sector_demand, combinations, days_by_cycle),
        current_assignment={(s, ((s - 1) % len(combinations)) + 1): 1 for s in sectors},
        max_churn=n_sectors,  # loose: the sweep isn't exercising the churn budget
        days_by_cycle=days_by_cycle,
    )


def run(
    solvers: list[str] = SOLVERS_TO_COMPARE,
    sector_sweep: list[int] = SECTOR_SWEEP,
    cycle_sweep: list[int] = CYCLE_SWEEP,
) -> list[SolveResult]:
    results = []
    for n_cycles in cycle_sweep:
        for n_sectors in sector_sweep:
            instance = generate_instance(n_sectors, n_cycles)
            for solver_name in solvers:
                result = run_block_assignment_mip(**instance, solver=solver_name)
                results.append(result)
                log_solve_result(
                    result,
                    params={"n_sectors": n_sectors, "n_cycles": n_cycles},
                    tags={"family": "mip", "n_sectors": str(n_sectors), "n_cycles": str(n_cycles)},
                )
    return results


def write_sweep_csv(
    results: list[SolveResult],
    sector_sweep: list[int],
    cycle_sweep: list[int],
    solvers: list[str],
    path: str | Path,
) -> None:
    """Like `write_results_csv`, plus the `n_sectors`/`n_cycles` each result swept over."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes = [
        (n_cycles, n_sectors)
        for n_cycles in cycle_sweep
        for n_sectors in sector_sweep
        for _ in solvers
    ]
    fieldnames = ["n_sectors", "n_cycles", *SolveResult.model_fields]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (n_cycles, n_sectors), result in zip(sizes, results, strict=True):
            writer.writerow({"n_sectors": n_sectors, "n_cycles": n_cycles, **result.model_dump()})


if __name__ == "__main__":
    # No-ops unless MLFLOW_TRACKING_URI is set (or configure_tracking is
    # called with an explicit uri) — see src/persistence/README.md.
    configure_tracking(experiment_name="compare_modeling")
    results = run()
    write_sweep_csv(results, SECTOR_SWEEP, CYCLE_SWEEP, SOLVERS_TO_COMPARE, OUTPUT_PATH)
    for result in results:
        print(result)
