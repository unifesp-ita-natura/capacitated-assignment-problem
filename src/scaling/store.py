"""Append-only CSV of solved sweep points, so an interrupted sweep can resume."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from src.scaling.sweep import Completed, Row


def load_completed(path: Path) -> Completed:
    """Rows already written to `path`, keyed by `(n_cycles, requested_sectors)`."""
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
    return {(int(row["n_cycles"]), int(row["requested_sectors"])): row for row in rows}


def append_row(path: Path, row: Row) -> None:
    """Add `row` to `path` (writing the header on first use), flushed to disk immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
