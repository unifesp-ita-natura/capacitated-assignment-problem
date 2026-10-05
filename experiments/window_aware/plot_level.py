"""Plot total items per cycle with the pooled LightGBM forecast of the next cycles and its band."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import FuncFormatter
from pydantic import TypeAdapter

import src.forecasting.candidates  # noqa: F401 - populate REGISTRY
from src.config.schema import ForecastParams
from src.forecasting.dataset import build_item_panel, load_demand_base
from src.forecasting.model import REGISTRY

CONFIG_PATH = "configs/experiments/window_aware/all.yaml"
CANDIDATE_LABEL = "razao + lag 19 + media anual"
ERRORS_CSV = "experiments/window_aware/outputs/errors.csv"
OUTPUT_PNG = "experiments/window_aware/outputs/level_forecast.png"
Z95 = 1.96


def future_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Rest of the current year's cycles, each opening 52 weeks after the same cycle last year.

    Sectors are the ones active in the last observed cycle. Each one is asked
    about the window it had in that cycle last year, 52 weeks later; a sector
    absent then opens with the cycle, for the cycle's median length.
    """
    last = panel["CICLOS"].max()
    year, number = int(last[:4]), int(last[4:])
    calendar = panel.drop_duplicates("CICLOS").set_index("CICLOS")["opening_date"]
    codes = [f"{year}{n:02d}" for n in range(number + 1, 20) if f"{year - 1}{n:02d}" in calendar]
    sectors = panel.loc[panel["CICLOS"] == last, "cd_setor"].unique()
    targets = pd.DataFrame(
        [
            {
                "cd_setor": s,
                "CICLOS": c,
                "year_ago": f"{year - 1}{c[4:]}",
                "opening_date": calendar[f"{year - 1}{c[4:]}"] + pd.Timedelta(weeks=52),
            }
            for c in codes
            for s in sectors
        ]
    )
    year_ago = panel.rename(columns={"CICLOS": "year_ago"})
    targets = targets.merge(
        year_ago[["cd_setor", "year_ago", "window_start", "cycle_days"]],
        on=["cd_setor", "year_ago"],
        how="left",
    )
    median_days = year_ago.groupby("year_ago")["cycle_days"].median()
    targets["window_start"] = (targets["window_start"] + pd.Timedelta(weeks=52)).fillna(
        targets["opening_date"]
    )
    targets["cycle_days"] = targets["cycle_days"].fillna(targets["year_ago"].map(median_days))
    return targets.drop(columns="year_ago")


def backtest_sigma() -> float:
    """Std of log(actual total / predicted total) across the backtest's one-step origins."""
    errors = pd.read_csv(ERRORS_CSV)
    totals = (
        errors[errors["candidate"] == CANDIDATE_LABEL]
        .groupby("CICLOS")[["actual", "items_pred"]]
        .sum()
    )
    return float(np.log(totals["actual"] / totals["items_pred"]).std(ddof=1))


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    entry = next(c for c in config["candidates"] if c.get("label") == CANDIDATE_LABEL)
    candidate = REGISTRY.build(TypeAdapter(ForecastParams).validate_python(entry))

    panel = build_item_panel(load_demand_base(config["paths"]["base_csv"]))
    targets = future_targets(panel)
    predicted = candidate.fit_predict(panel, targets)

    hist = panel.groupby("opening_date")["items"].sum() / 1e6
    fc = (
        predicted.merge(targets.drop_duplicates("CICLOS")[["CICLOS", "opening_date"]], on="CICLOS")
        .groupby(["opening_date", "CICLOS"])["items_pred"]
        .sum()
        .reset_index()
    )
    # Anchor on the last observed cycle, so the dashed line starts where the solid ends.
    fc_x = pd.DatetimeIndex([hist.index[-1], *fc["opening_date"]])
    mid = np.r_[hist.iloc[-1], fc["items_pred"].to_numpy() / 1e6]
    h = np.arange(len(mid))
    sigma = backtest_sigma()
    # ponytail: horizon-1 backtest error grown by sqrt(h) (random-walk assumption);
    # a multi-horizon backtest would measure the growth instead of assuming it.
    width = Z95 * sigma * np.sqrt(h)
    upper, lower = mid * np.exp(width), mid * np.exp(-width)
    print(f"sigma(log)={sigma:.4f}")
    print(fc.assign(lower=lower[1:] * 1e6, upper=upper[1:] * 1e6).to_string(index=False))

    plot(hist, fc_x, mid, lower, upper, fc["CICLOS"].iloc[-1], sigma)


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


def plot(hist, fc_x, mid, lower, upper, last_code, sigma) -> None:
    ink, grey = "#1a1a1a", "#6b6b6b"
    c_hist, c_fc = "#1f4e9c", "#d95f02"  # blue / orange: CVD-safe pair
    plt.rcParams.update({"font.size": 16, "font.family": "DejaVu Sans"})

    fig, ax = plt.subplots(figsize=(13, 7), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x0, pad = fc_x[0], pd.Timedelta(days=30)
    ax.axvspan(x0, fc_x[-1] + pad, color="#f2f2f2", zorder=0)
    ax.axvline(x0, color=grey, lw=2, zorder=1)
    ax.grid(axis="y", color="#dddddd", lw=1)
    ax.set_axisbelow(True)

    ax.fill_between(
        fc_x,
        lower,
        upper,
        color=c_fc,
        alpha=0.25,
        lw=0,
        zorder=2,
        label="Intervalo de previsão 95%",
    )
    ax.plot(fc_x, upper, color=c_fc, lw=1.5, ls=(0, (2, 2)), zorder=3)
    ax.plot(fc_x, lower, color=c_fc, lw=1.5, ls=(0, (2, 2)), zorder=3)
    ax.plot(
        hist.index,
        hist.values,
        color=c_hist,
        lw=3,
        marker="o",
        ms=5,
        zorder=4,
        label="Histórico (soma dos setores)",
    )
    ax.plot(
        fc_x,
        mid,
        color=c_fc,
        lw=3,
        ls=(0, (5, 2.5)),
        marker="o",
        ms=5,
        zorder=4,
        label="Previsão LightGBM (esperada)",
    )

    annotate_ends(
        ax,
        fc_x[-1],
        [
            (mid[-1], c_fc, 20, "bold"),
            (upper[-1], grey, 15, "normal"),
            (lower[-1], grey, 15, "normal"),
        ],
    )

    ticks = pd.date_range(hist.index[0].to_period("Q").to_timestamp(), fc_x[-1], freq="QS")
    ax.set_xticks(ticks)
    ax.set_xticklabels([d.strftime("%b %Y") for d in ticks])
    ax.set_xlim(hist.index[0] - pd.Timedelta(days=10), fc_x[-1] + pad)
    lo, hi = min(hist.min(), lower.min()), max(hist.max(), upper.max())
    span = hi - lo
    ax.set_ylim(0, hi + 0.15 * span)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 1)))
    ax.set_ylabel("Itens por ciclo (milhões)", color=ink, fontsize=17, labelpad=10)
    ax.tick_params(colors=ink, labelsize=15, length=6, width=1.5)
    style_spines(ax)

    ax.text(
        x0 + pd.Timedelta(days=4),
        ax.get_ylim()[1] - 0.02 * span,
        "Início da previsão",
        color=grey,
        fontsize=15,
        va="top",
        ha="left",
    )
    leg = ax.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=14,
        handlelength=2.6,
        borderpad=0.8,
    )
    for tx in leg.get_texts():
        tx.set_color(ink)

    fig.suptitle(
        f"Demanda deve fechar em ~{br(mid[-1], 1)} mi de itens no ciclo {last_code}",
        x=0.06,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color=ink,
    )
    fig.text(
        0.06,
        0.015,
        f"base_tratada_v2, um ponto por abertura de ciclo. Banda = previsão × exp(±1,96·σ·√h), "
        f"σ = {br(sigma, 3)} (erro log do backtest a 1 ciclo).",
        fontsize=11,
        color=grey,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUTPUT_PNG, facecolor="white")


if __name__ == "__main__":
    main()
