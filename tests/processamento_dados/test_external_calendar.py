"""Tests for the pure, offline calendar feature module (holidays, paydays, Bolsa Familia window)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.processamento_dados.external_calendar import (
    _national_holidays as national_holidays,
)
from src.processamento_dados.external_calendar import (
    _nth_sunday_of_month,
    _validate_estado,
    bolsa_familia_window,
    build_daily_calendar_features,
    is_bolsa_familia_window,
    is_business_day,
    is_dia_das_maes,
    is_dia_dos_namorados,
    is_dia_dos_pais,
    is_natal,
    is_national_holiday,
    is_quinto_dia_util,
    is_state_only_holiday,
    quinto_dia_util,
    summarize_cycle_calendar_features,
)
from src.processamento_dados.external_calendar import (
    _state_holidays as state_holidays,
)


def test_validate_estado_rejects_non_uf_codes():
    assert _validate_estado("sp") == "SP"
    with pytest.raises(ValueError):
        _validate_estado("SPX")


def test_dia_das_maes_is_second_sunday_of_may_2026():
    assert _nth_sunday_of_month(2026, 5, 2) == date(2026, 5, 10)
    assert is_dia_das_maes(date(2026, 5, 10))
    assert not is_dia_das_maes(date(2026, 5, 3))
    assert not is_dia_das_maes(date(2026, 5, 17))


def test_dia_dos_pais_is_second_sunday_of_august_2026():
    assert _nth_sunday_of_month(2026, 8, 2) == date(2026, 8, 9)
    assert is_dia_dos_pais(date(2026, 8, 9))
    assert not is_dia_dos_pais(date(2026, 8, 2))


def test_fixed_commemorative_dates():
    assert is_dia_dos_namorados(date(2026, 6, 12))
    assert not is_dia_dos_namorados(date(2026, 6, 13))
    assert is_natal(date(2026, 12, 25))
    assert not is_natal(date(2026, 12, 24))


def test_national_holiday_flags_known_2026_dates():
    nat = national_holidays([2026])
    assert is_national_holiday(date(2026, 1, 1), nat)
    assert is_national_holiday(date(2026, 4, 21), nat)
    assert not is_national_holiday(date(2026, 1, 2), nat)


def test_state_only_holiday_excludes_national_holidays():
    nat = national_holidays([2026])
    sp = state_holidays("SP", [2026])

    # Constitutionalist Revolution Day (Jul 9) is SP-specific, not national.
    assert is_state_only_holiday(date(2026, 7, 9), nat, sp)
    # Tiradentes (Apr 21) is national, so it must not be double-counted as state-only.
    assert not is_state_only_holiday(date(2026, 4, 21), nat, sp)
    assert not is_state_only_holiday(date(2026, 7, 10), nat, sp)


def test_is_business_day_excludes_weekends_and_national_holidays():
    nat = national_holidays([2026])
    assert is_business_day(date(2026, 3, 2), nat)  # Monday, no holiday
    assert not is_business_day(date(2026, 3, 7), nat)  # Saturday
    assert not is_business_day(date(2026, 1, 1), nat)  # national holiday (also a Thursday)


def test_quinto_dia_util_march_2026():
    nat = national_holidays([2026])

    assert quinto_dia_util(2026, 3, nat) == date(2026, 3, 6)
    assert is_quinto_dia_util(date(2026, 3, 6), nat)
    assert not is_quinto_dia_util(date(2026, 3, 5), nat)
    assert not is_quinto_dia_util(date(2026, 3, 9), nat)


def test_bolsa_familia_window_march_2026_spans_18th_to_31st():
    nat = national_holidays([2026])

    window = bolsa_familia_window(2026, 3, nat)

    assert len(window) == 10
    assert min(window) == date(2026, 3, 18)
    assert max(window) == date(2026, 3, 31)


def test_is_bolsa_familia_window_march_2026_boundaries():
    nat = national_holidays([2026])

    assert is_bolsa_familia_window(date(2026, 3, 18), nat)
    assert is_bolsa_familia_window(date(2026, 3, 31), nat)
    assert not is_bolsa_familia_window(date(2026, 3, 17), nat)


def test_build_daily_calendar_features_shape_and_columns():
    dates = pd.to_datetime(["2026-07-09", "2026-05-10"])
    estados = ["SP", "RJ"]

    df = build_daily_calendar_features(dates, estados)

    assert len(df) == len(dates) * len(estados)
    assert set(df.columns) == {
        "date",
        "estado",
        "is_national_holiday",
        "is_dia_das_maes",
        "is_dia_dos_namorados",
        "is_dia_dos_pais",
        "is_natal",
        "is_quinto_dia_util",
        "is_bolsa_familia_window",
        "is_state_holiday",
    }


def test_build_daily_calendar_features_varies_by_state():
    dates = pd.to_datetime(["2026-07-09", "2026-05-10"])
    estados = ["SP", "RJ"]

    df = build_daily_calendar_features(dates, estados)

    jul9_sp = df[(df["date"] == pd.Timestamp("2026-07-09")) & (df["estado"] == "SP")].iloc[0]
    jul9_rj = df[(df["date"] == pd.Timestamp("2026-07-09")) & (df["estado"] == "RJ")].iloc[0]
    assert bool(jul9_sp["is_state_holiday"])
    assert not bool(jul9_rj["is_state_holiday"])
    # National-only feature must agree across estados for the same date.
    assert bool(jul9_sp["is_national_holiday"]) == bool(jul9_rj["is_national_holiday"])

    maes_row = df[(df["date"] == pd.Timestamp("2026-05-10")) & (df["estado"] == "SP")].iloc[0]
    assert bool(maes_row["is_dia_das_maes"])


def test_summarize_cycle_calendar_features_holiday_counts_for_april_2026_cycle():
    # A 21-day cycle opening 2026-04-01 spans Good Friday (Apr 3) and
    # Tiradentes (Apr 21), both national holidays, and no SP-only holiday.
    summary = summarize_cycle_calendar_features("SP", date(2026, 4, 1), 21)

    assert summary["n_national_holidays"] == 2
    assert summary["n_state_holidays"] == 0


def test_summarize_cycle_calendar_features_commemorative_counts_for_april_2026_cycle():
    # The same cycle contains no Mother's/Valentine's/Father's Day or Christmas.
    summary = summarize_cycle_calendar_features("SP", date(2026, 4, 1), 21)

    assert summary["n_dia_das_maes"] == 0
    assert summary["n_dia_dos_namorados"] == 0
    assert summary["n_dia_dos_pais"] == 0
    assert summary["n_natal"] == 0


def test_summarize_cycle_calendar_features_window_counts_for_april_2026_cycle():
    # The quinto dia util (Apr 8) and 3 Bolsa Familia window days (Apr 16,
    # 17, 20) fall inside the cycle too (independently verified above).
    summary = summarize_cycle_calendar_features("SP", date(2026, 4, 1), 21)

    assert summary["has_quinto_dia_util"] is True
    assert summary["n_bolsa_familia_window_days"] == 3


def test_summarize_cycle_calendar_features_accepts_pandas_timestamp_open_date():
    summary = summarize_cycle_calendar_features("SP", pd.Timestamp("2026-04-01"), 21)

    assert summary["n_national_holidays"] == 2
