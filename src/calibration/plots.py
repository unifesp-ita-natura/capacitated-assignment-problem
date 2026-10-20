"""Gráficos PNG da calibração: convergência (energia e temperatura) e efeitos principais."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # só arquivos PNG; funciona mesmo depois de importar o pyplot

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
LINE_WIDTH = 2.0
FIGURE_DPI = 150
ENERGY_WINDOW = (0.98, 1.25)  # recorte do eixo de energia, relativo à menor energia


def _style_axis(ax: plt.Axes, ylabel: str) -> None:
    """Eixos discretos: grade horizontal clara, sem bordas superior/direita."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_convergence(traces: Mapping[str, pd.DataFrame], path: Path, title: str) -> None:
    """Melhor energia por iteração (painel de cima) e temperatura (painel de baixo, log)."""
    fig, (ax_energy, ax_temp) = plt.subplots(2, 1, sharex=True, figsize=(9, 6.5))
    fig.patch.set_facecolor(SURFACE)
    for color, (label, trace) in zip(SERIES_COLORS, traces.items(), strict=False):
        ax_energy.plot(
            trace["iteration"], trace["best_energy"], color=color, lw=LINE_WIDTH, label=label
        )
        ax_temp.plot(trace["iteration"], trace["temperature"], color=color, lw=LINE_WIDTH)
    _style_axis(ax_energy, "melhor energia E(X)")
    _style_axis(ax_temp, "temperatura T (log)")
    ax_energy.set_ylim(*_energy_window(traces))
    ax_temp.set_yscale("log")
    ax_temp.set_xlabel("iteração", color=MUTED)
    ax_energy.legend(frameon=False, labelcolor=TEXT)
    ax_energy.set_title(title, color=TEXT, loc="left")
    _save(fig, path)


def _energy_window(traces: Mapping[str, pd.DataFrame]) -> tuple[float, float]:
    """Faixa do eixo de energia perto do fim da busca.

    O As-Is inviável começa em ~rho x P_cap, ordens de grandeza acima da energia
    final; sem o recorte, a parte que diferencia as configurações vira uma linha.
    """
    floor = min(trace["best_energy"].min() for trace in traces.values())
    return ENERGY_WINDOW[0] * floor, ENERGY_WINDOW[1] * floor


def plot_main_effects(effects: pd.DataFrame, path: Path, title: str) -> None:
    """Pequenos múltiplos: gap médio por faixa de cada parâmetro (uma série por painel)."""
    params = list(dict.fromkeys(effects["param"]))
    fig, axes = plt.subplots(1, len(params), figsize=(3.2 * len(params), 3.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, param in zip(_as_list(axes), params, strict=True):
        _plot_effect(ax, effects[effects["param"] == param], param)
    _as_list(axes)[0].set_ylabel("gap médio (%)", color=MUTED)
    fig.suptitle(title, color=TEXT, x=0.01, ha="left")
    _save(fig, path)


def _plot_effect(ax: plt.Axes, effect: pd.DataFrame, param: str) -> None:
    positions = range(len(effect))
    ax.plot(positions, effect["gap_mean"], color=SERIES_COLORS[0], lw=LINE_WIDTH, marker="o", ms=5)
    ax.set_xticks(list(positions), effect["level"], rotation=30, ha="right", fontsize=8)
    ax.set_title(param, color=TEXT, fontsize=9)
    _style_axis(ax, "")


def _as_list(axes) -> list:
    return list(axes) if hasattr(axes, "__iter__") else [axes]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, facecolor=SURFACE)
    plt.close(fig)
