"""Tests for the early-stop, resumable size sweep."""

from __future__ import annotations

from src.persistence import SolveResult
from src.scaling.store import append_row, load_completed
from src.scaling.sweep import Trial, sweep, sweep_sectors


def _trial(condition: str, n_sectors: int) -> Trial:
    result = SolveResult(
        model_name="m",
        solver="s",
        status="ok",
        termination_condition=condition,
        objective=None,
        wall_time_seconds=1.0,
        git_commit=None,
    )
    return Trial(result, n_sectors)


def _runner(timeouts_from: int, available: int = 1000):
    def run_point(n_cycles: int, n_sectors: int) -> Trial:
        condition = "maxTimeLimit" if n_sectors >= timeouts_from else "optimal"
        return _trial(condition, min(n_sectors, available))

    return run_point


def test_stops_after_two_timeouts_and_skips_larger_sectors() -> None:
    rows = sweep_sectors(1, [40, 80, 120, 160, 200], _runner(timeouts_from=80), 2)
    assert [row["requested_sectors"] for row in rows] == [40, 80, 120]
    assert [row["timed_out"] for row in rows] == [False, True, True]


def test_runs_every_sector_when_nothing_times_out() -> None:
    rows = sweep_sectors(1, [40, 80, 120], _runner(timeouts_from=10_000), 2)
    assert len(rows) == 3


def test_stops_when_data_has_fewer_sectors_than_requested() -> None:
    rows = sweep_sectors(1, [40, 80, 120], _runner(10_000, available=60), 2)
    assert [row["requested_sectors"] for row in rows] == [40, 80]


def test_each_cycle_count_restarts_its_own_sweep() -> None:
    rows = sweep([1, 2], [40, 80, 120, 160], _runner(timeouts_from=80), 2)
    assert [(row["n_cycles"], row["requested_sectors"]) for row in rows] == [
        (1, 40),
        (1, 80),
        (1, 120),
        (2, 40),
        (2, 80),
        (2, 120),
    ]


def _recording_runner(timeouts_from: int):
    calls: list[tuple[int, int]] = []
    inner = _runner(timeouts_from)

    def run_point(n_cycles: int, n_sectors: int) -> Trial:
        calls.append((n_cycles, n_sectors))
        return inner(n_cycles, n_sectors)

    return run_point, calls


def test_resume_skips_completed_points_but_keeps_their_timeouts() -> None:
    rows: list[dict] = []
    first_runner, _ = _recording_runner(80)
    sweep_sectors(1, [40, 80, 120], first_runner, 2, on_row=rows.append)
    done = {(r["n_cycles"], r["requested_sectors"]): r for r in rows[:2]}

    second_runner, second_calls = _recording_runner(80)
    resumed = sweep_sectors(1, [40, 80, 120], second_runner, 2, done)

    assert second_calls == [(1, 120)]
    assert [r["requested_sectors"] for r in resumed] == [40, 80, 120]


def test_store_round_trips_rows(tmp_path) -> None:
    path = tmp_path / "scaling.csv"
    rows: list[dict] = []
    sweep_sectors(1, [40, 80], _runner(10_000), 2, on_row=rows.append)
    for row in rows:
        append_row(path, row)

    loaded = load_completed(path)

    assert set(loaded) == {(1, 40), (1, 80)}
    assert not loaded[(1, 40)]["timed_out"]
    assert loaded[(1, 40)]["objective"] is None
