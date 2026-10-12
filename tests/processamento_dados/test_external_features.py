"""Tests for the external-feature join logic (calendar + macro) onto the by-state panels."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from src.processamento_dados.external_features import (
    _quarter_period,
    _shape_calendar_date,
    add_calendar_features_to_level,
    add_calendar_features_to_shape,
    add_macro_features,
    drop_out_of_cycle_window_rows,
)


def _level_row(cd_setor=1, ciclo=202601, estado="SP", date="2026-01-05", cycle_duration=21):
    return pd.DataFrame(
        [
            {
                "cd_setor": cd_setor,
                "ciclo": ciclo,
                "estado": estado,
                "date": pd.Timestamp(date),
                "cycle_duration": cycle_duration,
            }
        ]
    )


def test_quarter_period_maps_month_to_the_right_quarter_code():
    assert _quarter_period(pd.Timestamp("2026-01-15")) == "202601"
    assert _quarter_period(pd.Timestamp("2026-06-30")) == "202602"
    assert _quarter_period(pd.Timestamp("2026-07-01")) == "202603"
    assert _quarter_period(pd.Timestamp("2025-12-05")) == "202504"


def test_shape_calendar_date_reconstructs_cycle_start_and_end():
    level = _level_row()
    shape = pd.DataFrame(
        [
            {"cd_setor": 1, "ciclo": 202601, "estado": "SP", "relative_date": 1 / 21},
            {"cd_setor": 1, "ciclo": 202601, "estado": "SP", "relative_date": 1.0},
        ]
    )

    dates = _shape_calendar_date(shape, level)

    assert dates.tolist() == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-25")]


def test_add_calendar_features_to_level_adds_rollup_columns():
    level = _level_row(date="2026-04-01", cycle_duration=21)

    result = add_calendar_features_to_level(level)

    assert result.loc[0, "n_national_holidays"] == 2  # Good Friday + Tiradentes
    assert result.loc[0, "has_quinto_dia_util"]


def test_drop_out_of_cycle_window_rows_removes_relative_dates_outside_0_1():
    shape = pd.DataFrame(
        [
            {"cd_setor": 1, "ciclo": 202601, "estado": "SP", "relative_date": 0.5},
            {"cd_setor": 1, "ciclo": 202601, "estado": "SP", "relative_date": -0.1},
            {"cd_setor": 1, "ciclo": 202601, "estado": "SP", "relative_date": 1.5},
        ]
    )

    result = drop_out_of_cycle_window_rows(shape)

    assert result["relative_date"].tolist() == [0.5]


def test_add_calendar_features_to_shape_drops_out_of_window_rows_by_default():
    level = _level_row(date="2026-01-05", cycle_duration=21)
    shape = pd.DataFrame(
        [
            {"cd_setor": 1, "ciclo": 202601, "estado": "SP", "relative_date": 1 / 21},
            {"cd_setor": 1, "ciclo": 202601, "estado": "SP", "relative_date": 1.5},
        ]
    )

    result = add_calendar_features_to_shape(shape, level)

    assert len(result) == 1
    assert result.loc[0, "relative_date"] == 1 / 21


def test_add_calendar_features_to_shape_keeps_out_of_window_rows_when_disabled():
    level = _level_row(date="2026-01-05", cycle_duration=21)
    shape = pd.DataFrame(
        [
            {"cd_setor": 1, "ciclo": 202601, "estado": "SP", "relative_date": 1 / 21},
            {"cd_setor": 1, "ciclo": 202601, "estado": "SP", "relative_date": 1.5},
        ]
    )

    result = add_calendar_features_to_shape(shape, level, drop_out_of_cycle_window=False)

    assert len(result) == 2


def test_add_macro_features_leaves_nan_for_unpublished_periods():
    frame = pd.DataFrame(
        [
            {"estado": "SP", "periodo": "202601"},
            {"estado": "SP", "periodo": "202699"},  # not a real/published period
        ]
    )
    macro_long = pd.DataFrame(
        [
            {"estado": "SP", "periodo": "202601", "indicador": "taxa_desocupacao", "valor": 5.0},
            {"estado": "SP", "periodo": "202601", "indicador": "rendimento_medio", "valor": 4000.0},
        ]
    )

    with patch(
        "src.processamento_dados.external_features.fetch_macro_indicators",
        return_value=macro_long,
    ):
        result = add_macro_features(frame)

    assert result.loc[0, "taxa_desocupacao"] == 5.0
    assert pd.isna(result.loc[1, "taxa_desocupacao"])
