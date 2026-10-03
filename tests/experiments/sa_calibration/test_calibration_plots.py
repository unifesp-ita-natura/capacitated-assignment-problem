"""Testes de fumaça dos gráficos da calibração (PNG gerado, sem checar pixels)."""

from __future__ import annotations

import pandas as pd
import pytest

# matplotlib vem do extra `viz`: sem ele, o módulo inteiro é pulado
plots = pytest.importorskip("src.calibration.plots")


def test_plot_convergence_writes_a_png(tmp_path):
    trace = pd.DataFrame(
        {
            "iteration": [0, 10, 20],
            "energy": [5.0, 4.0, 3.0],
            "best_energy": [5.0, 4.0, 3.0],
            "temperature": [100.0, 10.0, 1.0],
        }
    )
    path = tmp_path / "convergence.png"

    plots.plot_convergence({"a": trace, "b": trace}, path, "título")

    assert path.stat().st_size > 0


def test_plot_main_effects_writes_a_png(tmp_path):
    effects = pd.DataFrame(
        {
            "param": ["x", "x", "y", "y"],
            "level": ["(0, 1]", "(1, 2]", "(0, 1]", "(1, 2]"],
            "gap_mean": [5.0, 3.0, 4.0, 4.5],
        }
    )
    path = tmp_path / "effects.png"

    plots.plot_main_effects(effects, path, "título")

    assert path.stat().st_size > 0
