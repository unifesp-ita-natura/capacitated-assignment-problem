"""Early-stop, resumable sweep over instance sizes that stops growing after repeated timeouts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import NamedTuple

from src.persistence import SolveResult

TIMEOUT_CONDITIONS = ("maxTimeLimit", "noSolution")


Row = dict[str, object]
Completed = Mapping[tuple[int, int], Row]


class Trial(NamedTuple):
    """One solve: the result plus how many sectors the instance really had."""

    result: SolveResult
    n_sectors: int


def is_timeout(result: SolveResult) -> bool:
    """Whether the solver stopped on its time limit (with or without an incumbent)."""
    return any(condition in result.termination_condition for condition in TIMEOUT_CONDITIONS)


def sweep_sectors(
    n_cycles: int,
    sector_sweep: Iterable[int],
    run_point: Callable[[int, int], Trial],
    max_timeouts: int,
    completed: Completed | None = None,
    on_row: Callable[[Row], None] = lambda row: None,
) -> list[Row]:
    """Rows for growing `sector_sweep` at fixed `n_cycles`, stopping at `max_timeouts` timeouts.

    Also stops once a point asks for more sectors than the data has, since larger
    requests would just repeat the same instance. Points already in `completed`
    (keyed by `(n_cycles, requested_sectors)`) are reused, not re-solved, but still
    count toward the stopping rules; each freshly solved row goes to `on_row`.
    """
    completed = completed or {}
    rows: list[Row] = []
    timeouts = 0
    for requested in sector_sweep:
        row = completed.get((n_cycles, requested)) or _solve_point(
            n_cycles, requested, run_point, on_row
        )
        rows.append(row)
        timeouts += bool(row["timed_out"])
        if _should_stop(row, requested, timeouts, max_timeouts):
            break
    return rows


def sweep(
    cycle_sweep: Iterable[int],
    sector_sweep: list[int],
    run_point: Callable[[int, int], Trial],
    max_timeouts: int,
    completed: Completed | None = None,
    on_row: Callable[[Row], None] = lambda row: None,
) -> list[Row]:
    """Run `sweep_sectors` for every cycle count and concatenate the rows."""
    return [
        row
        for n_cycles in cycle_sweep
        for row in sweep_sectors(n_cycles, sector_sweep, run_point, max_timeouts, completed, on_row)
    ]


def _should_stop(row: Row, requested: int, timeouts: int, max_timeouts: int) -> bool:
    """Too many timeouts so far, or the data ran out of sectors for this request."""
    return timeouts >= max_timeouts or int(row["n_sectors"]) < requested  # type: ignore[call-overload]


def _solve_point(
    n_cycles: int,
    requested: int,
    run_point: Callable[[int, int], Trial],
    on_row: Callable[[Row], None],
) -> Row:
    trial = run_point(n_cycles, requested)
    row = _row(n_cycles, requested, trial, is_timeout(trial.result))
    on_row(row)
    return row


def _row(n_cycles: int, requested: int, trial: Trial, timed_out: bool) -> Row:
    return {
        "n_cycles": n_cycles,
        "requested_sectors": requested,
        "n_sectors": trial.n_sectors,
        "timed_out": timed_out,
        **trial.result.model_dump(),
    }
