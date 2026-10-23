"""Plots comparing as-is vs to-be daily load leveling against capacity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def plot_daily_leveling(ax, leveling_comparison: pd.DataFrame, day_capacity: float) -> None:
    """Plot as-is vs to-be daily captacao against the target capacity line."""
    ax.plot(
        leveling_comparison["order_date"], leveling_comparison["as_is"], marker="o", label="As-is"
    )
    ax.plot(
        leveling_comparison["order_date"], leveling_comparison["to_be"], marker="o", label="To-be"
    )
    ax.axhline(day_capacity, color="black", linestyle="--", label="Target capacity")
    ax.set_title("Daily captacao: as-is vs to-be")
    ax.set_ylabel("Forecast orders")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()


def plot_sectors_per_slot(
    ax, sectors_per_slot: pd.Series, sectors_per_slot_to_be: pd.Series, slots: list[tuple]
) -> None:
    """Plot the sector count per (block, subloco) slot, as-is vs to-be -- one bar pair per slot.

    Unlike a per-block count, this keeps every slot visible so a concentration hiding
    inside a single subloco (not just a single block) is still easy to spot.
    """
    width = 0.4
    x = np.arange(len(slots))
    ax.bar(x - width / 2, sectors_per_slot.values, width, label="As-is")
    ax.bar(x + width / 2, sectors_per_slot_to_be.values, width, label="To-be")
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{b}-S{s}" for b, s in slots], rotation=90, fontsize=8)
    ax.set_title("Sectors per (block, subloco) slot: as-is vs to-be")
    ax.set_ylabel("Number of sectors")
    ax.legend()


def plot_forecast_vs_actual(ax, forecast_accuracy: pd.DataFrame) -> None:
    """Plot forecast vs actual daily orders for the held-out validation cycle."""
    ax.plot(
        forecast_accuracy["order_date"],
        forecast_accuracy["forecast_orders"],
        marker="o",
        label="Forecast",
    )
    ax.plot(
        forecast_accuracy["order_date"],
        forecast_accuracy["actual_orders"],
        marker="o",
        label="Actual",
    )
    ax.set_title("Requests by day: forecast vs actual (as-is calendar)")
    ax.set_ylabel("Orders")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()


def plot_leveling_metrics(ax, leveling_metrics: pd.DataFrame) -> None:
    """Plot mean/std/peak daily load, as-is vs to-be, as a grouped bar chart."""
    metrics_to_plot = leveling_metrics.loc[["mean_daily_load", "std_daily_load", "peak_daily_load"]]
    metrics_to_plot[["as_is", "to_be"]].plot(kind="bar", ax=ax)
    ax.set_title("Load-leveling metrics: as-is vs to-be")
    ax.set_ylabel("Orders")
    ax.tick_params(axis="x", rotation=20)
