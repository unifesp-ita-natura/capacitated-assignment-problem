"""Tests for the lag/rolling/calendar feature table the pooled candidate trains on."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.features import (
    LEVEL_REFERENCE,
    build_features,
    complete_feature_frame,
    feature_columns,
)

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
).assign(window_start=lambda frame: frame["opening_date"], cycle_days=21)


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


def test_complete_frame_keeps_only_rows_with_every_feature_present():
    complete = complete_feature_frame(PANEL, LAGS, WINDOWS)

    # With lag_2 and a 2-cycle rolling mean, only cycles 3 and 4 qualify,
    # for each of the two sectors.
    assert len(complete) == 4
    assert complete[feature_columns(LAGS, WINDOWS)].notna().all().all()


def test_complete_frame_pools_every_sector_into_one_matrix():
    complete = complete_feature_frame(PANEL, LAGS, WINDOWS)

    assert set(complete["cd_setor"]) == {"A", "B"}


# Same panel, plus the order and volume counts `build_item_panel` now
# carries. A's items-per-order ratio is deliberately uneven (10, 5, 10, 20)
# so a lagged ratio can't be confused with a constant.
PANEL_WITH_COUNTS = PANEL.assign(
    orders=[10, 40, 30, 20, 1, 2, 3, 4],
    volumes=[5, 6, 7, 8, 1, 1, 1, 1],
)
COMPANION_LAGS = [1, 2]


def test_companion_lags_are_absent_unless_asked_for():
    # Default behaviour must be byte-for-byte what it was before order
    # counts reached the panel, so earlier results stay reproducible.
    featured = build_features(PANEL_WITH_COUNTS, LAGS, WINDOWS)

    assert not [column for column in featured.columns if column.startswith("orders_lag_")]


def test_companion_lags_hold_the_previous_cycles_counts():
    featured = build_features(PANEL_WITH_COUNTS, LAGS, WINDOWS, COMPANION_LAGS).set_index(
        ["cd_setor", "CICLOS"]
    )

    assert featured.loc[("A", "202604"), "orders_lag_1"] == 30
    assert featured.loc[("A", "202604"), "orders_lag_2"] == 40
    assert featured.loc[("A", "202604"), "volumes_lag_1"] == 7


def test_companion_lags_come_from_the_same_sector_only():
    featured = build_features(PANEL_WITH_COUNTS, LAGS, WINDOWS, COMPANION_LAGS).set_index(
        ["cd_setor", "CICLOS"]
    )

    assert pd.isna(featured.loc[("B", "202601"), "orders_lag_1"])
    assert featured.loc[("B", "202602"), "orders_lag_1"] == 1


def test_items_per_order_lag_is_an_earlier_cycles_ratio_never_its_own():
    featured = build_features(PANEL_WITH_COUNTS, LAGS, WINDOWS, COMPANION_LAGS).set_index(
        ["cd_setor", "CICLOS"]
    )

    # Cycle 3's ratio is 300/30 = 10; cycle 4's own ratio (400/20 = 20)
    # must never appear in cycle 4's features.
    assert featured.loc[("A", "202604"), "items_per_order_lag_1"] == 10
    assert featured.loc[("A", "202604"), "items_per_order_lag_2"] == 5


def test_feature_columns_list_the_companion_features_when_asked():
    columns = feature_columns(LAGS, WINDOWS, COMPANION_LAGS)

    assert "orders_lag_1" in columns
    assert "volumes_lag_2" in columns
    assert "items_per_order_lag_1" in columns
    assert not any("BLOCO" in column.upper() for column in columns)


def test_complete_frame_carries_the_columns_a_target_is_cut_from():
    complete = complete_feature_frame(PANEL_WITH_COUNTS, LAGS, WINDOWS, COMPANION_LAGS)

    columns = feature_columns(LAGS, WINDOWS, COMPANION_LAGS)
    assert not complete[columns].isna().to_numpy().any()
    # It keeps the non-feature columns a caller cuts a target from.
    assert {"items", "CICLOS", LEVEL_REFERENCE} <= set(complete.columns)


def test_level_reference_is_the_running_mean_of_earlier_cycles_only():
    featured = build_features(PANEL_WITH_COUNTS, LAGS, WINDOWS).set_index(["cd_setor", "CICLOS"])

    # A's earlier cycles are 100, 200, 300 -> 200. Its own 400 is excluded,
    # which is what makes this usable as a ratio denominator.
    assert featured.loc[("A", "202604"), LEVEL_REFERENCE] == 200
    assert pd.isna(featured.loc[("A", "202601"), LEVEL_REFERENCE])


def test_organization_features_are_opt_in():
    assert "CD_RE" not in feature_columns(LAGS, WINDOWS)
    assert {"CD_RE", "CD_GV"} <= set(feature_columns(LAGS, WINDOWS, (), ["CD_RE", "CD_GV"]))


def test_missing_organization_codes_do_not_remove_training_rows():
    panel = PANEL.assign(CD_RE=pd.NA)
    complete = complete_feature_frame(panel, LAGS, WINDOWS, categorical_features=["CD_RE"])
    assert len(complete) == 4


def test_requested_organization_column_must_exist():
    with pytest.raises(ValueError, match="missing categorical features.*CD_GV"):
        complete_feature_frame(PANEL, LAGS, WINDOWS, categorical_features=["CD_GV"])
