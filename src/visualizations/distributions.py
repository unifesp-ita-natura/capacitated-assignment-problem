"""Generic annotated bar/histogram helpers for exploring count and share distributions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns


def _annotate_bars(ax, fmt) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if height <= 0:
            continue
        ax.annotate(
            fmt(height),
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def plot_percentage_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, order=None):
    """Bar chart of a share series, annotated with each bar's percentage."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        x=series.index,
        y=series.values,
        order=order,
        color="#2b5c8f",
        edgecolor="white",
        alpha=0.85,
        ax=ax,
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    _annotate_bars(ax, lambda h: f"{h * 100:.1f}%")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, ylim=(0, series.max() * 1.15))
    fig.tight_layout()
    return fig, ax


def plot_count_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, order=None):
    """Bar chart of a count series, annotated with each bar's integer value."""
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        x=series.index,
        y=series.values,
        order=order,
        color="#2b5c8f",
        edgecolor="white",
        alpha=0.85,
        ax=ax,
    )
    _annotate_bars(ax, lambda h: f"{int(h)}")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, ylim=(0, series.max() * 1.1))
    fig.tight_layout()
    return fig, ax


def plot_stacked_percentage_bar(wide_shares: pd.DataFrame, title: str, xlabel: str, ylabel: str):
    """Stacked bar chart of per-group shares, annotated with each segment's percentage."""
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("viridis", n_colors=wide_shares.shape[1])
    bottom = pd.Series(0.0, index=wide_shares.index)
    for column, color in zip(wide_shares.columns, palette):
        values = wide_shares[column]
        ax.bar(
            wide_shares.index,
            values,
            bottom=bottom,
            color=color,
            edgecolor="white",
            label=str(column),
        )
        for row, value, base in zip(wide_shares.index, values, bottom):
            if value >= 0.015:
                ax.annotate(
                    f"{value * 100:.1f}%",
                    xy=(row, base + value / 2),
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )
        bottom += values
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set(
        title=title, xlabel=xlabel, ylabel=ylabel, ylim=(0, wide_shares.sum(axis=1).max() * 1.15)
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_density_histogram(series: pd.Series, title: str, xlabel: str):
    """Discrete density histogram with KDE and mean/median reference lines."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        series,
        kde=True,
        stat="density",
        discrete=True,
        color="#2b5c8f",
        edgecolor="white",
        alpha=0.6,
        ax=ax,
    )
    median_val, mean_val = series.median(), series.mean()
    ax.axvline(
        median_val,
        color="darkorange",
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {median_val:.1f}",
    )
    ax.axvline(
        mean_val, color="crimson", linestyle=":", linewidth=1.5, label=f"Mean: {mean_val:.1f}"
    )
    ax.set(title=title, xlabel=xlabel, ylabel="Density")
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig, ax


def plot_top_n_barh(series: pd.Series, title: str, xlabel: str, ylabel: str, n: int = 15):
    """Horizontal bar chart of the top-n series values, labeled with exact counts."""
    top = series.nlargest(n).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index.astype(str), top.values, color="#2b5c8f", edgecolor="white", alpha=0.85)
    for bar in ax.patches:
        ax.text(
            bar.get_width() + max(top.values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width())}",
            va="center",
            fontsize=10,
        )
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    return fig, ax
