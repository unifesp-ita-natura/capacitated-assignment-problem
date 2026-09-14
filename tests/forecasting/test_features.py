"""Tests for the lag/rolling/calendar feature table the pooled candidate trains on."""

from __future__ import annotations

import pandas as pd

from src.forecasting.features import build_features, build_training_table, feature_columns

LAGS = [1, 2]
WINDOWS = [2]

PANEL = pd.DataFrame(
    {
        "cd_setor": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "CICLOS": ["202601", "202602", "202603", "202604"] * 2,
        "items": [100, 200, 300, 400, 10, 20, 30, 40],
        "opening_date": pd.to_datetime(
            ["2026-01-05", "2026-02-02", "2026-03-02", "2026-04-02"] * 2
        ),
    }
)


def test_lags_come_from_the_same_sector_only():
    featured = build_features(PANEL, LAGS, WINDOWS).set_index(["cd_setor", "CICLOS"])

    # B's first cycle has no prior cycle of its own — it must not inherit
    # A's last value just because A's rows come first in the frame.
    assert pd.isna(featured.loc[("B", "202601"), "lag_1"])
    assert featured.loc[("B", "202602"), "lag_1"] == 10


def test_lag_columns_hold_the_previous_cycles_values():
    featured = build_features(PANEL, LAGS, WINDOWS).set_index(["cd_setor", "CICLOS"])

    assert featured.loc[("A", "202604"), "lag_1"] == 300
    assert featured.loc[("A", "202604"), "lag_2"] == 200


def test_rolling_mean_excludes_the_row_being_predicted():
    featured = build_features(PANEL, LAGS, WINDOWS).set_index(["cd_setor", "CICLOS"])

    # This is the leakage guard: the mean for cycle 4 averages cycles 2 and
    # 3 (200, 300) — never cycle 4's own 400.
    assert featured.loc[("A", "202604"), "rolling_mean_2"] == 250


def test_calendar_features_come_from_the_cycle_not_the_block():
    featured = build_features(PANEL, LAGS, WINDOWS).set_index(["cd_setor", "CICLOS"])

    assert featured.loc[("A", "202603"), "cycle_number"] == 3
    assert featured.loc[("A", "202603"), "opening_month"] == 3


def test_feature_columns_never_include_block_assignment():
    # Standing team rule: block/sub-block is the optimizer's decision
    # variable and must never be a predictor (see docs/agent-log).
    columns = feature_columns(LAGS, WINDOWS)

    assert not any("BLOCO" in column.upper() for column in columns)
    assert not any("bloco" in column.lower() for column in columns)


def test_training_table_keeps_only_rows_with_every_feature_present():
    features, target = build_training_table(PANEL, LAGS, WINDOWS)

    # With lag_2 and a 2-cycle rolling mean, only cycles 3 and 4 qualify,
    # for each of the two sectors.
    assert len(features) == 4
    assert len(target) == 4
    assert features.notna().all().all()


def test_training_table_pools_every_sector_into_one_matrix():
    features, _ = build_training_table(PANEL, LAGS, WINDOWS)

    assert set(features["cd_setor"]) == {"A", "B"}
