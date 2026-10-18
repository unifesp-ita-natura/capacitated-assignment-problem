"""Solve the block-assignment model across a sector/cycle size sweep with each
configured solver, and write a comparison table.

Capacity is kept deliberately loose (far above any possible demand) so the
sweep isolates how solve time/behavior scales with instance size, rather than
with capacity feasibility.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from pathlib import Path

from src.persistence import SolveResult, configure_tracking, log_solve_result
from src.solver.inputs import ProblemInstance, ProjectedDemand
from src.solver.mip.block_assignment import run_block_assignment_mip

# Add/remove Pyomo-registered solver names to compare different backends
# against the same model. To compare different *formulations* instead (or in
# addition), swap the generated instance below for a list of
# instances/builders and loop over both.
SOLVERS_TO_COMPARE = ["gurobi", "appsi_highs"]

OUTPUT_PATH = "experiments/compare_modeling/outputs/results.csv"

# How many sectors the synthetic instance has, swept from a small toy size up
# to a large one.
SECTOR_SWEEP = list(range(40, 1000, 200))

# How many forecast cycles the horizon spans, crossed with SECTOR_SWEEP.
CYCLE_SWEEP = [1, 2, 4, 8]

# Each cycle's length (in days) is sampled independently (with replacement)
# from this range, so cycles aren't all the same size.
CYCLE_LENGTH_RANGE = (19, 23)

# Combinations = the sector's choice of which day its own campaign starts
# on (combination `c` starts a new campaign on day `c`, every cycle).
# Fixed regardless of cycle length; must not exceed CYCLE_LENGTH_RANGE's minimum, so every
# combination has a valid start day in every cycle.
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


def _day_position_lookup(days_by_cycle: Mapping[int, list[int]]) -> dict[int, tuple[int, int]]:
    """Global day -> (1-indexed position within its own cycle, that cycle's length)."""
    return {
        day: (offset + 1, len(cycle_days))
        for cycle_days in days_by_cycle.values()
        for offset, day in enumerate(cycle_days)
    }


def _combination_profile_value(
    day: int, combination: int, day_position_lookup: Mapping[int, tuple[int, int]]
) -> float:
    """Fraction (in [0, 1]) of a sector's demand landing on `day` under `combination`.

    Combination `c` starts a new campaign on day `c` of every global cycle
    (`day - (c - 1)` shifts `day` back onto that combination's own start-day
    frame, then looks up its position/length within the global cycle
    structure — see the day-21/day-4 walkthroughs in the PR description for
    why this shift, not a plain day-of-cycle lookup, is correct).

    Full demand right when the campaign launches, declining linearly to 0 by
    the last day of that cycle (before the next campaign starts). Since
    every one of the `N_COMBINATIONS` combinations contributes on every day
    (just at a different point in its own campaign's decline), no day's
    *aggregate* demand collapses to 0 the way a single-spike-per-cycle model
    would.
    """
    shifted_day = day - (combination - 1)
    if shifted_day < 1:
        return 0.0
    position = day_position_lookup.get(shifted_day)
    if position is None:
        return 0.0
    p, length = position
    if length == 1:
        return 1.0
    return max(0.0, 1.0 - (p - 1) / (length - 1))


def _build_projected_demand(
    sector_demand: Mapping[int, float],
    combinations: list[int],
    days_by_cycle: Mapping[int, list[int]],
) -> ProjectedDemand:
    """Each sector's demand follows `_combination_profile_value`'s decline curve
    under every combination it could be assigned, for every day in the horizon.
    """
    day_position_lookup = _day_position_lookup(days_by_cycle)
    return {
        (s, day, combination): demand
        * _combination_profile_value(day, combination, day_position_lookup)
        for s, demand in sector_demand.items()
        for day in day_position_lookup
        for combination in combinations
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
) -> ProblemInstance:
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

    return ProblemInstance(
        sectors=sectors,
        combinations=combinations,
        days=days,
        cd_sectors=cd_sectors,
        daily_capacity=_build_loose_daily_capacity(sector_demand, cd_sectors, days),
        projected_demand=_build_projected_demand(sector_demand, combinations, days_by_cycle),
        current_assignment={s: ((s - 1) % len(combinations)) + 1 for s in sectors},
        max_churn=n_sectors,  # loose: the sweep isn't exercising the churn budget
        days_by_cycle=days_by_cycle,
    )


ResultSink = Callable[[int, int, SolveResult], None]


CompletedRun = tuple[int, int, str]  # (n_sectors, n_cycles, solver)


def _completed_runs(path: str | Path) -> set[CompletedRun]:
    """`(n_sectors, n_cycles, solver)` combos already written to `path`.

    Lets a restarted sweep skip work it already has a row for. Note `solver`
    here is the *resolved* solver (see `SolveResult.solver`): on a machine
    with no Gurobi license, every "gurobi" request falls back to and is
    recorded as "appsi_highs", so its slot in `SOLVERS_TO_COMPARE` won't be
    recognized as completed on restart and gets (harmlessly) redone.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with open(path, newline="") as f:
        return {
            (int(row["n_sectors"]), int(row["n_cycles"]), row["solver"])
            for row in csv.DictReader(f)
        }


@contextmanager
def open_sweep_csv(path: str | Path, overwrite: bool = False) -> Generator[ResultSink]:
    """A sink that appends+flushes one row per call, instead of buffering the
    whole sweep in memory and writing it out at the end.

    A full sweep can run for hours (some solves hit the time limit); writing
    only on completion would mean a crash or interrupt partway through loses
    every result computed so far. Appends (rather than truncating) and skips
    the header when `path` already has rows, so re-running after a crash or
    `kill` resumes instead of wiping prior progress — see `_completed_runs`,
    which `run`'s `completed` argument should be seeded from on restart.
    `overwrite=True` (the `--overwrite` CLI flag) truncates `path` and starts
    fresh instead, for a deliberate from-scratch run. Yields a callable to
    pass as `run`'s `on_result`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["n_sectors", "n_cycles", *SolveResult.model_fields]
    write_header = overwrite or not path.exists() or path.stat().st_size == 0
    with open(path, "w" if overwrite else "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        def write_result(n_sectors: int, n_cycles: int, result: SolveResult) -> None:
            writer.writerow({"n_sectors": n_sectors, "n_cycles": n_cycles, **result.model_dump()})
            f.flush()

        yield write_result


def _pending_solvers(
    solvers: list[str], n_sectors: int, n_cycles: int, completed: set[CompletedRun]
) -> list[str]:
    return [
        solver_name
        for solver_name in solvers
        if (n_sectors, n_cycles, solver_name) not in completed
    ]


def _run_one(
    n_sectors: int,
    n_cycles: int,
    solver_name: str,
    instance: ProblemInstance,
    on_result: ResultSink | None,
) -> SolveResult:
    result = run_block_assignment_mip(instance, solver=solver_name)
    log_solve_result(
        result,
        params={"n_sectors": n_sectors, "n_cycles": n_cycles},
        tags={"family": "mip", "n_sectors": str(n_sectors), "n_cycles": str(n_cycles)},
    )
    if on_result is not None:
        on_result(n_sectors, n_cycles, result)
    return result


def _run_instance(
    n_sectors: int,
    n_cycles: int,
    solvers: list[str],
    completed: set[CompletedRun],
    on_result: ResultSink | None,
) -> list[SolveResult]:
    """Every pending solver's result for one `(n_sectors, n_cycles)` instance.

    Builds the instance only if at least one solver for it isn't already in
    `completed`, so a resumed sweep doesn't pay to regenerate instances it
    won't use.
    """
    pending_solvers = _pending_solvers(solvers, n_sectors, n_cycles, completed)
    if not pending_solvers:
        return []
    instance = generate_instance(n_sectors, n_cycles)
    return [
        _run_one(n_sectors, n_cycles, solver_name, instance, on_result)
        for solver_name in pending_solvers
    ]


def run(
    solvers: list[str] = SOLVERS_TO_COMPARE,
    sector_sweep: list[int] = SECTOR_SWEEP,
    cycle_sweep: list[int] = CYCLE_SWEEP,
    on_result: ResultSink | None = None,
    completed: set[CompletedRun] | None = None,
) -> list[SolveResult]:
    completed = completed or set()
    return [
        result
        for n_cycles in cycle_sweep
        for n_sectors in sector_sweep
        for result in _run_instance(n_sectors, n_cycles, solvers, completed, on_result)
    ]


def _report(n_sectors: int, n_cycles: int, result: SolveResult, write_result: ResultSink) -> None:
    write_result(n_sectors, n_cycles, result)
    print(n_sectors, n_cycles, result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Ignore any existing results.csv and start the sweep from scratch "
            "(truncating the file), instead of resuming by skipping already-"
            "completed (n_sectors, n_cycles, solver) runs."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # No-ops unless MLFLOW_TRACKING_URI is set (or configure_tracking is
    # called with an explicit uri) — see src/persistence/README.md.
    configure_tracking(experiment_name="compare_modeling")
    completed = set() if args.overwrite else _completed_runs(OUTPUT_PATH)
    if completed:
        print(f"Resuming: {len(completed)} (n_sectors, n_cycles, solver) runs already done.")
    with open_sweep_csv(OUTPUT_PATH, overwrite=args.overwrite) as write_result:
        run(
            completed=completed,
            on_result=lambda n_sectors, n_cycles, result: _report(
                n_sectors, n_cycles, result, write_result
            ),
        )
