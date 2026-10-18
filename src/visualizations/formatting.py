"""Generic matplotlib number-formatting and styling helpers shared across chart scripts."""

from __future__ import annotations

import pandas as pd


def br(value: float, decimals: int) -> str:
    """Number with a decimal comma, as the client reads it."""
    return f"{value:.{decimals}f}".replace(".", ",")


def annotate_ends(ax, last_x, labels) -> None:
    """Print each (value, color, size, weight) at the right end of its line."""
    end_x = last_x + pd.Timedelta(days=5)
    for value, color, size, weight in labels:
        ax.annotate(
            br(value, 2),
            (last_x, value),
            xytext=(end_x, value),
            va="center",
            ha="left",
            color=color,
            fontsize=size,
            fontweight=weight,
        )


def style_spines(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
