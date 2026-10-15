"""Generate the 'Curva de shape' slide figure: one sector's curve and the lambda/janela sweep.

Left panel: Método 1 (média histórica) vs Método 2 (ponderada por recência) vs o
valor observado, para um setor com histórico completo, no ciclo de validação
mais recente. Right panel: MAE da shape (métrica principal do slide) por lambda,
uma linha por tamanho de janela (3, 6, 12 ciclos), medido com o backtest real
(src.forecasting.shape_evaluation) sobre toda a base.

Eixo x normalizado em [0, 1] (posição relativa dentro do ciclo, bin*n_bins//n_days),
não o dia absoluto do ciclo -- ciclos têm 7 a 31 dias, então um dia absoluto não é
comparável entre setores/ciclos de tamanhos diferentes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.forecasting.dataset import CYCLE_KEYS, load_demand_base
from src.forecasting.shape_daily import build_shape_days, predict_shape
from src.forecasting.shape_evaluation import (
    ShapeCandidate,
    ShapeSplit,
    compare_shapes,
    evaluate_shapes,
)

BASE_CSV = "data/base_tratada_v2.csv"
OUTPUT_PNG = "experiments/shape_curve_slide/outputs/shape_curve.png"
N_BINS = 10
BASELINE_DECAY = 1.0
HIGHLIGHT_DECAY = 0.8
SWEEP_DECAYS = [0.8, 0.9, 1.0]
SWEEP_WINDOWS = [3, 6, 12]
C_BASELINE, C_RECENCY, C_ACTUAL = "#1f4e9c", "#d95f02", "#1a1a1a"


def _pick_sector(days: pd.DataFrame) -> str:
    """Pick the sector with full cycle history and the largest total volume."""
    complete = days.groupby("cd_setor")["CICLOS"].nunique()
    full_history = complete[complete.eq(complete.max())].index
    totals = days[days["cd_setor"].isin(full_history)].groupby("cd_setor")["actual"].sum()
    return totals.idxmax()


def _bin_actual(days: pd.DataFrame, sector: str, cycle: str) -> pd.DataFrame:
    """Bin one sector-cycle's observed shares the same way predictions are binned."""
    target = days[days["cd_setor"].eq(sector) & days["CICLOS"].eq(cycle)].copy()
    target["bin"] = target["offset"] * N_BINS // target["n_days"]
    return target.groupby("bin", as_index=False)["actual_share"].sum()


def _bin_forecast(forecast: pd.DataFrame, window_start, n_days: int) -> pd.Series:
    """Sum a daily share forecast into the same relative-position bins as the actual curve."""
    offset = (forecast["date"] - window_start).dt.days
    bins = offset * N_BINS // n_days
    return forecast["share_pred"].groupby(bins).sum().reindex(range(N_BINS), fill_value=0)


def _curve_panel(ax, days: pd.DataFrame, sector: str) -> None:
    """Plot Método 1 vs Método 2 vs the actual curve for one sector's last cycle."""
    cycles = days[days["cd_setor"].eq(sector)].sort_values("opening_date")["CICLOS"].unique()
    history, target_cycle = days[days["CICLOS"].isin(cycles[:-1])], cycles[-1]
    window = days[days["cd_setor"].eq(sector) & days["CICLOS"].eq(target_cycle)][
        [*CYCLE_KEYS, "window_start", "n_days"]
    ].drop_duplicates()
    window_start, n_days = window["window_start"].iloc[0], window["n_days"].iloc[0]

    x = (pd.Series(range(N_BINS)) + 0.5) / N_BINS
    for name, model, decay, color in [
        ("Método 1: média histórica", "sector", BASELINE_DECAY, C_BASELINE),
        (f"Método 2: recência (λ={HIGHLIGHT_DECAY})", "recency", HIGHLIGHT_DECAY, C_RECENCY),
    ]:
        forecast = predict_shape(history, window, model, N_BINS, decay)
        binned = _bin_forecast(forecast, window_start, n_days)
        ax.plot(x, binned, marker="o", lw=2.5, color=color, label=name)

    actual = (
        _bin_actual(days, sector, target_cycle)
        .set_index("bin")
        .reindex(range(N_BINS), fill_value=0)
    )
    ax.plot(x, actual["actual_share"], marker="s", lw=2, ls="--", color=C_ACTUAL, label="Observado")
    ax.set_title(f"Setor {sector}, ciclo {target_cycle} (validação)")
    ax.set_xlabel("Posição relativa no ciclo (0-1)")
    ax.set_ylabel("Fração do volume do ciclo")
    ax.legend(fontsize=9)


def _sweep(days: pd.DataFrame) -> pd.DataFrame:
    """Backtest share MAE for every (janela, lambda) combination on the whole base."""
    rows = []
    for window in SWEEP_WINDOWS:
        split = ShapeSplit(min_train_cycles=window, window="sliding")
        candidates = [
            ShapeCandidate(name=f"lambda={decay}", model="recency", n_bins=N_BINS, decay=decay)
            for decay in SWEEP_DECAYS
        ]
        comparison = compare_shapes(evaluate_shapes(days, candidates, split)).set_index("candidate")
        for decay in SWEEP_DECAYS:
            rows.append(
                dict(
                    window=window,
                    decay=decay,
                    mae_pp=comparison.loc[f"lambda={decay}", "share_mae_pp"],
                )
            )
    return pd.DataFrame(rows)


def _sweep_panel(ax, sweep: pd.DataFrame) -> None:
    """Plot MAE da shape (pp) by lambda, one line per janela."""
    for window, group in sweep.groupby("window"):
        group = group.sort_values("decay")
        ax.plot(
            group["decay"], group["mae_pp"], marker="o", lw=2.5, label=f"janela={window} ciclos"
        )
    ax.set_title("MAE da shape por λ e janela (backtest, base completa)")
    ax.set_xlabel("λ (peso de recência)")
    ax.set_ylabel("MAE da shape (p.p.)")
    ax.legend(fontsize=9)


def style_spines(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")


def main() -> None:
    days = build_shape_days(load_demand_base(BASE_CSV))
    sector = _pick_sector(days)
    sweep = _sweep(days)
    sweep.to_csv("experiments/shape_curve_slide/outputs/lambda_window_sweep.csv", index=False)
    print(sweep.to_string(index=False))

    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    fig.patch.set_facecolor("white")
    for ax in (left, right):
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#dddddd", lw=1)
        ax.set_axisbelow(True)
        style_spines(ax)

    _curve_panel(left, days, sector)
    _sweep_panel(right, sweep)

    fig.suptitle("Curva de shape: média histórica vs. ponderação por recência", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_PNG, facecolor="white")
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
