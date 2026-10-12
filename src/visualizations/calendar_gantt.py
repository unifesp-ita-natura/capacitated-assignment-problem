"""Gantt-style plot of sector sales windows on the validation-cycle calendar."""

from __future__ import annotations

import matplotlib.dates as mdates
import pandas as pd


def plot_calendar_gantt(
    ax,
    assignment: dict,
    cycle_open_date,
    sector_order: list,
    title: str,
    block_colors: dict,
    slot_start_day,
    window_length: int,
) -> None:
    """Draw each sector's sales window as a horizontal bar on the actual calendar."""
    for row, sector in enumerate(sector_order):
        block, subloco = assignment[sector]
        start_day = slot_start_day(block, subloco)
        start_date = cycle_open_date + pd.Timedelta(days=start_day - 1)
        ax.broken_barh(
            [(start_date, pd.Timedelta(days=window_length))],
            (row - 0.4, 0.8),
            facecolors=block_colors[block],
            edgecolors="white",
        )
    ax.set_yticks(range(len(sector_order)))
    ax.set_yticklabels(sector_order, fontsize=7)
    ax.set_ylim(-1, len(sector_order))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
