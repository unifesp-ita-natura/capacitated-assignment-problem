"""Janelas deslizantes sobre a lista ordenada de CDs (Relax-and-Fix e Fix-and-Optimize)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RelaxAndFixWindow:
    """Uma iteração do Relax-and-Fix: que CDs estão fixos, binários e relaxados."""

    fixed: tuple[int, ...]  # C_fix: já processados, valores congelados
    interior: tuple[int, ...]  # C_int: janela de W CDs, variáveis binárias
    relaxed: tuple[int, ...]  # C_relax: ainda não processados, x em [0, 1]
    step: tuple[int, ...]  # C_step ⊆ C_int: congelados no fim da iteração


def relax_and_fix_windows(cds: Sequence[int], window: int, step: int) -> list[RelaxAndFixWindow]:
    """Sequência de janelas do Relax-and-Fix sobre `cds`, com tamanho `window` e passo `step`.

    Cada iteração congela só os `step` primeiros CDs da janela, então com
    `step < window` janelas consecutivas se sobrepõem (o que ajuda a manter a
    viabilidade). Na última janela, que já não tem CD relaxado, todos os CDs
    restantes são congelados.
    """
    _validate_windows(len(cds), window, step)
    windows = []
    start = 0
    while True:
        interior = tuple(cds[start : start + window])
        relaxed = tuple(cds[start + window :])
        frozen_now = interior if not relaxed else interior[:step]
        windows.append(RelaxAndFixWindow(tuple(cds[:start]), interior, relaxed, frozen_now))
        if not relaxed:
            return windows
        start += step


def widen_window(window: RelaxAndFixWindow, undo: int) -> RelaxAndFixWindow:
    """Desfaz o congelamento dos `undo` últimos grupos fixos (backtracking do Relax-and-Fix).

    Esses grupos voltam pra janela (binários) e pro passo, pra ficarem congelados de novo
    no fim da iteração. `undo` maior que o número de grupos fixos desfaz todos.
    """
    cut = len(window.fixed) - min(undo, len(window.fixed))
    kept, undone = window.fixed[:cut], window.fixed[cut:]
    return RelaxAndFixWindow(kept, undone + window.interior, window.relaxed, undone + window.step)


def fix_and_optimize_windows(cds: Sequence[int], window: int, step: int) -> list[tuple[int, ...]]:
    """Janelas de CDs a liberar, uma por vez, no Fix-and-Optimize.

    Desliza de `step` em `step`; se o passo não encaixa exatamente no fim da
    lista, acrescenta uma última janela alinhada ao fim pra todo CD ser coberto.
    """
    _validate_windows(len(cds), window, step)
    starts = range(0, max(len(cds) - window, 0) + 1, step)
    windows = [tuple(cds[start : start + window]) for start in starts]
    if starts[-1] + window < len(cds):
        windows.append(tuple(cds[-window:]))
    return windows


def _validate_windows(n_cds: int, window: int, step: int) -> None:
    if n_cds < 1:
        raise ValueError("é preciso ao menos um CD")
    if window < 1:
        raise ValueError(f"window deve ser >= 1, recebi {window}")
    if not 1 <= step <= window:
        raise ValueError(f"step deve estar entre 1 e window ({window}), recebi {step}")
