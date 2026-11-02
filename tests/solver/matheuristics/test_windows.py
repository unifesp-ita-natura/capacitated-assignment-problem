"""Testes das janelas de CDs do Relax-and-Fix e do Fix-and-Optimize."""

from __future__ import annotations

import pytest

from src.solver.matheuristics.windows import (
    RelaxAndFixWindow,
    fix_and_optimize_windows,
    relax_and_fix_windows,
    widen_window,
)

CDS = [1, 2, 3, 4, 5, 6]


def test_relax_and_fix_windows_match_the_documented_example():
    # |C| = 6, W = 3, P = 2 (o exemplo dos slides da apresentação).
    windows = relax_and_fix_windows(CDS, window=3, step=2)

    assert windows == [
        RelaxAndFixWindow(fixed=(), interior=(1, 2, 3), relaxed=(4, 5, 6), step=(1, 2)),
        RelaxAndFixWindow(fixed=(1, 2), interior=(3, 4, 5), relaxed=(6,), step=(3, 4)),
        RelaxAndFixWindow(fixed=(1, 2, 3, 4), interior=(5, 6), relaxed=(), step=(5, 6)),
    ]


def test_relax_and_fix_with_overlap_reoptimizes_the_cds_left_unfrozen():
    windows = relax_and_fix_windows(CDS, window=3, step=2)

    # O CD 3 entra na janela da 1a e da 2a iteração (efeito da sobreposição P < W).
    assert 3 in windows[0].interior and 3 in windows[1].interior
    assert 3 not in windows[0].step


def test_relax_and_fix_without_overlap_partitions_the_cds():
    windows = relax_and_fix_windows(CDS, window=2, step=2)

    assert [w.interior for w in windows] == [(1, 2), (3, 4), (5, 6)]
    assert [w.step for w in windows] == [(1, 2), (3, 4), (5, 6)]


def test_relax_and_fix_with_window_larger_than_the_cds_is_a_single_window():
    windows = relax_and_fix_windows([1, 2], window=5, step=2)

    assert windows == [RelaxAndFixWindow(fixed=(), interior=(1, 2), relaxed=(), step=(1, 2))]


def test_every_cd_is_frozen_exactly_once_when_no_cd_repeats_in_step():
    windows = relax_and_fix_windows(CDS, window=3, step=2)

    frozen = [cd for w in windows for cd in w.step]
    assert sorted(frozen) == CDS


def test_fix_and_optimize_windows_slide_by_the_step():
    assert fix_and_optimize_windows(CDS, window=3, step=1) == [
        (1, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
        (4, 5, 6),
    ]


def test_fix_and_optimize_adds_a_tail_window_when_the_step_does_not_fit():
    windows = fix_and_optimize_windows(CDS, window=3, step=2)

    assert windows == [(1, 2, 3), (3, 4, 5), (4, 5, 6)]


@pytest.mark.parametrize(("window", "step"), [(0, 1), (3, 0), (2, 3), (-1, 1)])
def test_invalid_window_parameters_are_rejected(window, step):
    with pytest.raises(ValueError):
        relax_and_fix_windows(CDS, window=window, step=step)
    with pytest.raises(ValueError):
        fix_and_optimize_windows(CDS, window=window, step=step)


def test_empty_cd_list_is_rejected():
    with pytest.raises(ValueError):
        relax_and_fix_windows([], window=1, step=1)


def _window() -> RelaxAndFixWindow:
    return RelaxAndFixWindow(fixed=(1, 2), interior=(3, 4), relaxed=(5,), step=(3,))


def test_widen_window_returns_the_last_frozen_groups_to_the_window_and_the_step():
    widened = widen_window(_window(), undo=1)

    assert widened == RelaxAndFixWindow(fixed=(1,), interior=(2, 3, 4), relaxed=(5,), step=(2, 3))


def test_widen_window_beyond_the_frozen_groups_undoes_all_of_them():
    widened = widen_window(_window(), undo=9)

    assert widened == RelaxAndFixWindow(
        fixed=(), interior=(1, 2, 3, 4), relaxed=(5,), step=(1, 2, 3)
    )


def test_widen_window_by_zero_changes_nothing():
    assert widen_window(_window(), undo=0) == _window()
