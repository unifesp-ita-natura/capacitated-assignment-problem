"""Tests for the common-subset comparison that keeps candidate rankings honest."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.comparison import common_scored_keys, compare
from src.forecasting.evaluation import EvaluationResult, OriginResult


def _result(name: str, rows: list[tuple[str, str, float, float]]) -> EvaluationResult:
    """Build an EvaluationResult directly from (sector, cycle, actual, predicted) rows."""
    scored = pd.DataFrame(
        [
            {
                "cd_setor": sector,
                "CICLOS": cycle,
                "actual": actual,
                "items_pred": predicted,
                "abs_error": abs(actual - predicted),
                "origin_cycle": "o1",
            }
            for sector, cycle, actual, predicted in rows
        ]
    )
    origin = OriginResult(
        origin_cycle="o1",
        target_cycles=("c1",),
        scored=scored,
        missing=pd.DataFrame(columns=["cd_setor", "CICLOS", "origin_cycle"]),
        naive_in_sample_mae=1.0,
    )
    return EvaluationResult(candidate_name=name, origins=[origin])


def test_common_keys_are_the_intersection_across_candidates():
    wide = _result("wide", [("A", "c1", 100, 90), ("B", "c1", 50, 40)])
    narrow = _result("narrow", [("A", "c1", 100, 95)])

    keys = common_scored_keys([wide, narrow])

    assert keys == {("A", "c1", "o1")}


def test_common_keys_are_empty_without_results():
    assert common_scored_keys([]) == set()


def test_comparison_scores_every_candidate_on_the_same_points():
    # `narrow` skipped sector B, whose error `wide` had to carry. Ranking on
    # each candidate's own coverage would reward the skipping.
    wide = _result("wide", [("A", "c1", 100, 90), ("B", "c1", 50, 0)])
    narrow = _result("narrow", [("A", "c1", 100, 80)])

    table = compare([wide, narrow]).set_index("candidate")

    assert table.loc["wide", "n_scored_own"] == 2
    assert table.loc["wide", "n_scored_common"] == 1
    assert table.loc["wide", "mae_common"] == 10  # sector A only
    assert table.loc["narrow", "mae_common"] == 20


def test_comparison_ranks_by_the_common_subset_not_by_own_coverage():
    # `wide` looks worse on its own numbers (it carries sector B's big miss)
    # but is the better candidate on the points both actually predicted.
    wide = _result("wide", [("A", "c1", 100, 90), ("B", "c1", 50, 0)])
    narrow = _result("narrow", [("A", "c1", 100, 80)])

    table = compare([wide, narrow])

    assert table["mae_own"].iloc[0] > table["mae_own"].iloc[1]  # ranked first despite worse own MAE
    assert table["candidate"].iloc[0] == "wide"


def test_comparison_rejects_an_empty_candidate_list():
    with pytest.raises(ValueError, match="nothing to compare"):
        compare([])


def test_a_candidate_that_scored_nothing_gets_a_nan_common_error():
    scored_nothing = EvaluationResult(candidate_name="empty", origins=[])
    wide = _result("wide", [("A", "c1", 100, 90)])

    table = compare([wide, scored_nothing]).set_index("candidate")

    assert pd.isna(table.loc["empty", "mae_common"])
