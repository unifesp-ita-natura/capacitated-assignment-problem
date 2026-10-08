"""Testes da agregação: gaps por execução, tabela por configuração e regra de decisão."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.calibration import analysis
from src.calibration.gap import INFEASIBLE_GAP, MipReference

REFS = {
    "loose": MipReference("loose", 100.0, "optimal", 0.01, 100.0),
    "tight": MipReference("tight", 80.0, "lower_bound", 0.2, 100.0),
}


def _runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "config_id": ["a", "a", "b", "b"],
            "instance": ["loose", "tight", "loose", "tight"],
            "objective": [110.0, 100.0, 105.0, 50.0],
            "feasible": [True, True, True, False],
            "iterations": [10, 10, 20, 20],
            "time": [1.0, 1.0, 2.0, 2.0],
            "initial_temperature": [1000.0, 1000.0, 50.0, 50.0],
            "cooling_rate": [0.99, 0.99, 0.9, 0.9],
            "min_temperature": [0.01, 0.01, 1.0, 1.0],
            "sector_bias": [0.8, 0.8, 0.8, 0.8],
            "destination_bias": [0.7, 0.7, 0.7, 0.7],
            "penalty_coefficient": [1e6, 1e6, 1e6, 1e6],
        }
    )


def test_add_gaps_uses_reference_and_incumbent():
    runs = analysis.add_gaps(_runs(), REFS)

    assert list(runs["gap"][:3]) == pytest.approx([10.0, 25.0, 5.0])
    assert runs["gap"].iloc[3] == INFEASIBLE_GAP
    assert runs["gap_vs_incumbent"].iloc[1] == pytest.approx(0.0)
    assert list(runs["mip_ref_kind"][:2]) == ["optimal", "lower_bound"]


def test_add_group_marks_tight_instances():
    runs = analysis.add_group(_runs(), ["tight"])

    assert list(runs["group"]) == ["loose", "tight", "loose", "tight"]


def test_summarize_aggregates_per_config():
    summary = analysis.summarize(analysis.add_gaps(_runs(), REFS)).set_index("config_id")

    assert summary.loc["a", "gap_mean"] == pytest.approx(17.5)
    assert summary.loc["b", "feasible_rate"] == 0.5
    assert summary.loc["b", "n_runs"] == 2


def test_complexity_counts_changes_from_the_repo_default():
    default = {"initial_temperature": 1000.0, "cooling_rate": 0.99, "min_temperature": 0.01}

    assert analysis.complexity(default) == 0
    assert analysis.complexity({**default, "initial_temperature": 50.0}) == 1


def _summary(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["config_id", "gap_mean", "gap_p90", "complexity"])


def test_rule_prefers_lowest_mean_gap():
    ranked = analysis.rank_by_rule(_summary([("a", 5.0, 9.0, 0), ("b", 2.0, 9.0, 3)]), 0.5)

    assert list(ranked["config_id"]) == ["b", "a"]


def test_rule_breaks_mean_ties_by_p90_then_simplicity():
    rows = [("a", 5.0, 9.0, 3), ("b", 5.2, 7.0, 3), ("c", 5.1, 7.1, 1)]

    ranked = analysis.rank_by_rule(_summary(rows), 0.5)

    assert list(ranked["config_id"]) == ["c", "b", "a"]


def test_rule_drops_configs_without_gap():
    ranked = analysis.rank_by_rule(_summary([("a", math.nan, 1.0, 0), ("b", 1.0, 1.0, 0)]), 0.5)

    assert list(ranked["config_id"]) == ["b"]


def test_main_effects_has_one_row_per_level_and_param():
    runs = analysis.add_gaps(_runs(), REFS)

    effects = analysis.main_effects(runs, ["initial_temperature"], n_bins=2)

    assert set(effects["param"]) == {"initial_temperature"}
    assert effects["n_runs"].sum() == len(runs)


def test_history_from_runs_pairs_configs_with_scores():
    runs = analysis.add_gaps(_runs(), REFS)

    history = dict(
        (c["initial_temperature"], s)
        for c, s in analysis.history_from_runs(runs, ["initial_temperature"])
    )

    assert history == pytest.approx({1000.0: 17.5, 50.0: 52.5})


def test_best_of_returns_the_top_config_and_its_params():
    runs = analysis.add_gaps(_runs(), REFS)

    best_id, params = analysis.best_of(runs, tolerance=0.5)

    assert best_id == "a"
    assert params["initial_temperature"] == 1000.0


def test_mip_table_shows_which_reference_is_used():
    records = [
        {"instance": "x", "termination": "optimal", "lower_bound": 9.0, "upper_bound": 10.0},
        {"instance": "y"},
    ]

    table = analysis.mip_table(records).set_index("instance")

    assert table.loc["x", "reference_kind"] == "optimal"
    assert table.loc["x", "reference"] == 10.0
    assert table.loc["y", "reference_kind"] == "missing"


def test_always_feasible_drops_configs_with_any_infeasible_run():
    kept = analysis.always_feasible(_runs())

    assert set(kept["config_id"]) == {"a"}


def test_feasibility_by_reports_the_rate_per_level():
    table = analysis.feasibility_by(_runs(), "initial_temperature", n_bins=2)

    assert list(table["feasible_rate"]) == [0.5, 1.0]
    assert table["n_runs"].sum() == 4


def test_interval_label_is_short_and_readable():
    assert analysis.interval_label(pd.Interval(0.000299999, 0.0181234)) == "0.0003–0.0181"
