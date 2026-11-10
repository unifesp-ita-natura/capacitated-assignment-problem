"""Parse a Gurobi log into the incumbent and best-bound trajectory over time."""

from __future__ import annotations

import re
from typing import NamedTuple

_NODE_LINE = re.compile(r"\s(\d+)s$")


class TracePoint(NamedTuple):
    """Gurobi's incumbent and best bound as of `seconds` into the solve."""

    seconds: float
    incumbent: float
    bound: float


def _to_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _parse_row(line: str) -> TracePoint | None:
    """A table row ends `incumbent bound gap It/Node time`; None for any other line."""
    if not _NODE_LINE.search(line) or "%" not in line:
        return None
    tokens = line.split()
    incumbent, bound = _to_float(tokens[-5]), _to_float(tokens[-4])
    if incumbent is None or bound is None:
        return None
    return TracePoint(float(tokens[-1].rstrip("s")), incumbent, bound)


def parse_trace(log_text: str) -> list[TracePoint]:
    """Read the branch-and-bound table rows of a Gurobi log."""
    rows = (_parse_row(line) for line in log_text.splitlines())
    return [row for row in rows if row is not None]


def state_at(trace: list[TracePoint], seconds: float) -> TracePoint | None:
    """The last logged state at or before `seconds`, or None if nothing was logged yet."""
    known = [p for p in trace if p.seconds <= seconds]
    return known[-1] if known else None
