"""Plots comparing cycle-total and within-window shape forecasting strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def plot_level_forecasts(
    ax, level_forecasts: dict[str, pd.DataFrame], actual_cycle_total: pd.DataFrame, sectors: list
) -> None:
    """Plot each strategy's forecast cycle total for a few sectors against the actual value."""
    width = 0.8 / (len(level_forecasts) + 1)
    x = np.arange(len(sectors))
    for i, (name, forecast_df) in enumerate(level_forecasts.items()):
        values = forecast_df.set_index("sector").loc[sectors, "forecast_cycle_total"]
        ax.bar(x + i * width, values, width, label=name)
    actual_values = actual_cycle_total.set_index("sector").loc[sectors, "actual_cycle_total"]
    ax.bar(x + len(level_forecasts) * width, actual_values, width, label="actual", color="black")
    ax.set_xticks(x + len(level_forecasts) * width / 2)
    ax.set_xticklabels(sectors)
    ax.set_title("Forecast cycle total by strategy (sample sectors)")
    ax.set_ylabel("Cycle total")
    ax.legend(fontsize=8)


def plot_level_strategy_scores(ax, level_scoreboard: pd.DataFrame) -> None:
    """Plot each level-forecast strategy's WMAPE against the held-out validation cycle."""
    level_scoreboard["cycle_total_WMAPE"].plot(kind="bar", ax=ax, color="tab:blue")
    ax.set_title("Strategy accuracy on the held-out cycle")
    ax.set_ylabel("Cycle-total WMAPE")
    ax.tick_params(axis="x", rotation=20)


def plot_shape_strategies(ax, shape_strategies: dict[str, pd.DataFrame], sector) -> None:
    """Plot each strategy's within-window shape curve for one sector."""
    for name, shape_df in shape_strategies.items():
        sector_shape = shape_df[shape_df["sector"].eq(sector)].sort_values("weekday_offset")
        ax.plot(sector_shape["weekday_offset"], sector_shape["order_share"], marker="o", label=name)
    ax.set_title(f"Sector {sector}: shape-forecast strategies")
    ax.set_xlabel("Day offset within window")
    ax.set_ylabel("Share of cycle volume")
    ax.legend(fontsize=8)


def plot_strategy_scores(ax, strategy_scoreboard: pd.DataFrame) -> None:
    """Plot each shape-forecast strategy's WMAPE against the held-out validation cycle."""
    strategy_scoreboard["daily_WMAPE"].plot(kind="bar", ax=ax, color="tab:blue")
    ax.set_title("Strategy accuracy on the held-out cycle")
    ax.set_ylabel("Daily WMAPE")
    ax.tick_params(axis="x", rotation=20)


def plot_within_window_shapes(ax, cycle_shape: pd.DataFrame, sectors: list) -> None:
    """Plot each sector's forecasted within-window order-share curve."""
    for sector in sectors:
        sector_shape = cycle_shape[cycle_shape["sector"].eq(sector)]
        ax.plot(
            sector_shape["weekday_offset"],
            sector_shape["order_share"],
            marker="o",
            label=f"Sector {sector}",
        )
    ax.set_title(f"Forecasted within-window shape (first {len(sectors)} sectors)")
    ax.set_xlabel("Day offset within the sector's window")
    ax.set_ylabel("Share of sector cycle volume")
    ax.legend(fontsize=8)


def plot_forecast_timeseries(ax, forecast_daily: pd.DataFrame) -> None:
    """Plot the as-is calendar's forecast order time series for the validation cycle."""
    ax.plot(
        forecast_daily["order_date"],
        forecast_daily["forecast_orders"],
        marker="o",
        color="tab:blue",
        label="Forecast orders (as-is calendar)",
    )
    ax.set_title("Forecast time series for validation cycle (as-is assignment)")
    ax.set_xlabel("Order date")
    ax.set_ylabel("Orders")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
