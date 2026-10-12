"""3D ridge plot comparing as-is vs to-be daily demand across sectors against capacity."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _ridge_polygon(days: np.ndarray, y0: float, z: np.ndarray) -> list[tuple[float, float, float]]:
    n_days = len(days)
    return list(zip(days, [y0] * n_days, z)) + list(zip(days[::-1], [y0] * n_days, [0] * n_days))


def _draw_sector_ridges(
    ax,
    curves: dict[str, np.ndarray],
    days: np.ndarray,
    sector_names: list[str],
    palette: list[str],
    depth_gap: float,
) -> None:
    for row, name in enumerate(sector_names):
        ax.add_collection3d(
            Poly3DCollection(
                [_ridge_polygon(days, row * depth_gap, curves[name])],
                facecolor=mcolors.to_rgba(palette[row], 0.6),
                edgecolor=palette[row],
                linewidth=1.3,
            )
        )


def _draw_total_ridge(ax, total: np.ndarray, days: np.ndarray, y_total: float) -> None:
    ax.add_collection3d(
        Poly3DCollection(
            [_ridge_polygon(days, y_total, total)],
            facecolor=mcolors.to_rgba("#333333", 0.45),
            edgecolor="#222222",
            linewidth=2.6,
        )
    )


def _draw_capacity_line(ax, days: np.ndarray, y_total: float, capacity: float) -> None:
    ax.plot(
        days,
        [y_total] * len(days),
        [capacity] * len(days),
        linestyle="--",
        color="#C0392B",
        linewidth=2.2,
    )


def _draw_peak_callout(
    ax,
    total: np.ndarray,
    days: np.ndarray,
    y_total: float,
    capacity: float,
    max_height: float,
) -> None:
    peak_idx = int(np.argmax(total))
    peak_day, peak_val = days[peak_idx], total[peak_idx]
    over_capacity = peak_val > capacity
    peak_color = "#C0392B" if over_capacity else "#2E8B57"
    symbol = ">" if over_capacity else "≤"
    # clear the higher of (peak, capacity line) so the callout never sits under the dashed line
    label_z = max(peak_val, capacity) + max_height * 0.09
    ax.text(
        peak_day,
        y_total,
        label_z,
        f"Pico: {peak_val:.0f} {symbol} {capacity:.0f}",
        fontsize=19,
        fontweight="bold",
        color=peak_color,
        ha="center",
    )


def _configure_panel_axes(
    ax,
    days: np.ndarray,
    sector_names: list[str],
    depth_gap: float,
    y_total: float,
    y_max: float,
    max_height: float,
    title: str,
) -> None:
    n_days = len(days)
    ax.set_xlim(0, n_days)
    ax.set_ylim(0, y_max)
    ax.set_zlim(0, max_height)
    ax.set_xticks(np.arange(5, n_days, 5))  # drop the "0" tick: it collides with "Setor A"
    ax.set_xlabel("Dia da campanha", fontsize=22, labelpad=18)
    ax.set_yticks([i * depth_gap for i in range(len(sector_names))] + [y_total])
    ax.set_yticklabels([s.split(" (")[0] for s in sector_names] + ["Total"], fontsize=18)
    ax.set_zlabel("Volume de pedidos", fontsize=22, labelpad=18)
    ax.tick_params(axis="x", labelsize=19, pad=4)
    ax.tick_params(axis="z", labelsize=19, pad=6)
    ax.set_title(title, fontsize=30, fontweight="bold", pad=18, y=0.97)
    ax.view_init(elev=26, azim=-50)
    ax.set_box_aspect(None, zoom=1.18)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.grid(False)


def draw_demand_ridge_panel(
    ax,
    curves: dict[str, np.ndarray],
    total: np.ndarray,
    days: np.ndarray,
    sector_names: list[str],
    palette: list[str],
    capacity: float,
    max_height: float,
    title: str,
) -> None:
    """Draw one as-is/to-be panel: one ridge per sector, a total ridge, and a capacity line."""
    ax.set_facecolor("white")
    depth_gap = 1.4
    y_total = len(sector_names) * depth_gap + 0.6
    y_max = y_total + 0.8

    _draw_sector_ridges(ax, curves, days, sector_names, palette, depth_gap)
    _draw_total_ridge(ax, total, days, y_total)
    _draw_capacity_line(ax, days, y_total, capacity)
    _draw_peak_callout(ax, total, days, y_total, capacity, max_height)
    _configure_panel_axes(ax, days, sector_names, depth_gap, y_total, y_max, max_height, title)


def plot_as_is_to_be_comparison(
    scenarios: dict[str, tuple[dict, np.ndarray]],
    days: np.ndarray,
    sector_names: list[str],
    palette: list[str],
    capacity: float,
):
    """Assemble the side-by-side as-is/to-be 3D demand ridge figure.

    `scenarios` maps a panel title to (curves_by_sector, total_curve).
    """
    max_height = (
        max(
            max(max(c.max() for c in curves.values()), total.max())
            for curves, total in scenarios.values()
        )
        * 1.3
    )

    fig = plt.figure(figsize=(22, 12.5), facecolor="white")
    for i, (title, (curves, total)) in enumerate(scenarios.items(), start=1):
        ax = fig.add_subplot(1, len(scenarios), i, projection="3d")
        draw_demand_ridge_panel(
            ax, curves, total, days, sector_names, palette, capacity, max_height, title
        )

    plt.subplots_adjust(left=0.03, right=0.97, top=1.0, bottom=0.08, wspace=0.35)
    return fig
