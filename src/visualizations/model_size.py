"""Line plots of MIP variable and constraint counts across the sector/cycle sweep."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.visualizations.formatting import style_spines


def _plot_metric(ax, sizes: pd.DataFrame, column: str, title: str) -> None:
    """One line per cycle count, coloured light to dark by how many cycles."""
    cycle_counts = sorted(sizes["n_cycles"].unique())
    colors = plt.cm.viridis([i / max(len(cycle_counts) - 1, 1) for i in range(len(cycle_counts))])
    for color, n_cycles in zip(colors, cycle_counts, strict=True):
        rows = sizes[sizes["n_cycles"] == n_cycles]
        ax.plot(rows["n_sectors"], rows[column], color=color, linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Sectors")
    ax.grid(alpha=0.25)
    style_spines(ax)


def plot_model_size(sizes: pd.DataFrame) -> Figure:
    """Variables and constraints vs sectors, one line per number of cycles.

    `sizes` needs columns n_sectors, n_cycles, variables, constraints.
    """
    fig, (ax_vars, ax_cons) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    _plot_metric(ax_vars, sizes, "variables", "Variables")
    _plot_metric(ax_cons, sizes, "constraints", "Constraints")
    cycle_counts = sorted(sizes["n_cycles"].unique())
    norm = plt.Normalize(cycle_counts[0], cycle_counts[-1])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=[ax_vars, ax_cons], pad=0.02
    )
    colorbar.set_label("Cycles")
    fig.suptitle("Block-assignment MIP size")
    return fig
