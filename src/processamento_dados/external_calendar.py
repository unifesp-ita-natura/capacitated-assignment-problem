"""Computes pure, offline calendar features (holidays, paydays, Bolsa Familia window) by date/UF.

Two public entry points:

- `build_daily_calendar_features(dates, estados)` — one row per (date, estado)
  with boolean flags for every feature below. National-level flags
  (`is_national_holiday`, `is_dia_das_maes`, `is_dia_dos_namorados`,
  `is_dia_dos_pais`, `is_natal`, `is_quinto_dia_util`,
  `is_bolsa_familia_window`) are repeated identically across every estado row
  for a given date; `is_state_holiday` varies per (date, estado). Meant to
  join onto `demanda_shape_by_state.csv` on (`data_pedido`, `estado`).
- `summarize_cycle_calendar_features(estado, open_date, cycle_duration)` — one
  row per cycle with counts/flags aggregated over the cycle's day window,
  meant to join onto `demanda_level_by_state.csv` on (`estado`, `date`
  [= open_date]).

Data sources, all offline:

1. National and state holidays come from the `holidays` package
   (`holidays.Brazil(years=..., subdiv=<UF>)`), which already covers national
   holidays plus all 27 UFs' state holidays. No web calls.
2. Fixed/floating commemorative dates (Dia das Maes, Dia dos Namorados, Dia
   dos Pais, Natal) are hand-coded date rules — Mother's/Father's Day are the
   2nd Sunday of May/August respectively, the other two are fixed calendar
   dates. Natal (Dec 25) is kept as its own flag even though it duplicates
   the national holiday date, because the two represent different mechanisms
   (a gift-giving occasion vs. a day off work) and downstream models may want
   them separately.
3. Quinto dia util (5th business day of the month) is computed from the
   national holiday calendar: a business day is any date that is not a
   Saturday/Sunday and not a national holiday.
4. The Bolsa Familia disbursement window is a **simplified, hand-coded
   approximation** of the real policy: payments are actually staggered across
   the last ~10 business days of each month by the final digit of each
   beneficiary's NIS number, with the exact day-to-digit mapping published
   monthly as a PDF on gov.br (no bulk machine-readable official source
   exists). Since sector-level NIS-digit distributions aren't available,
   this module only flags "this date falls within the last 10 business days
   of its month" as a single national window, not an exact per-digit payday.
   Treat `is_bolsa_familia_window` as a coarse proxy, not an exact payday.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import holidays
import pandas as pd

_BOLSA_FAMILIA_WINDOW_BUSINESS_DAYS = 10
_QUINTO_DIA_UTIL_ORDINAL = 5


def _validate_estado(estado: str) -> str:
    """Raises if `estado` is not a 2-letter UF code, otherwise returns it uppercased."""
    normalized = estado.strip().upper()
    if len(normalized) != 2 or not normalized.isalpha():
        raise ValueError(f"estado must be a 2-letter UF code, got {estado!r}")
    return normalized


def _to_date(value: date | pd.Timestamp) -> date:
    """Normalizes a date-like value (datetime.date or pandas Timestamp) to `datetime.date`."""
    if isinstance(value, pd.Timestamp):
        return value.date()
    return value


def _national_holidays(years: Sequence[int]) -> holidays.HolidayBase:
    """Returns the Brazil national holiday calendar for the given years."""
    return holidays.Brazil(years=list(years))


def _state_holidays(estado: str, years: Sequence[int]) -> holidays.HolidayBase:
    """Returns the Brazil national+state holiday calendar for one UF and the given years."""
    return holidays.Brazil(years=list(years), subdiv=_validate_estado(estado))


def is_national_holiday(d: date, national_calendar: holidays.HolidayBase) -> bool:
    """Returns whether `d` is a Brazilian national holiday, per a precomputed calendar."""
    return d in national_calendar


def is_state_only_holiday(
    d: date,
    national_calendar: holidays.HolidayBase,
    state_calendar: holidays.HolidayBase,
) -> bool:
    """Returns whether `d` is a holiday in `state_calendar` but not already a national one."""
    return d in state_calendar and d not in national_calendar


def is_business_day(d: date, national_calendar: holidays.HolidayBase) -> bool:
    """Returns whether `d` is a business day: not Sat/Sun and not a national holiday."""
    return d.weekday() < 5 and not is_national_holiday(d, national_calendar)


def _nth_sunday_of_month(year: int, month: int, n: int) -> date:
    """Returns the date of the n-th Sunday of the given year/month (n is 1-indexed)."""
    first_of_month = date(year, month, 1)
    days_to_first_sunday = (6 - first_of_month.weekday()) % 7
    first_sunday = first_of_month + timedelta(days=days_to_first_sunday)
    return first_sunday + timedelta(weeks=n - 1)


def is_dia_das_maes(d: date) -> bool:
    """Returns whether `d` is Dia das Maes (2nd Sunday of May)."""
    return d == _nth_sunday_of_month(d.year, 5, 2)


def is_dia_dos_pais(d: date) -> bool:
    """Returns whether `d` is Dia dos Pais (2nd Sunday of August)."""
    return d == _nth_sunday_of_month(d.year, 8, 2)


def is_dia_dos_namorados(d: date) -> bool:
    """Returns whether `d` is Dia dos Namorados (fixed June 12)."""
    return d.month == 6 and d.day == 12


def is_natal(d: date) -> bool:
    """Returns whether `d` is Natal (fixed December 25)."""
    return d.month == 12 and d.day == 25


def _business_days_in_month(
    year: int, month: int, national_calendar: holidays.HolidayBase
) -> list[date]:
    """Returns every business day in the given year/month, in ascending order."""
    next_month_first = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    days_in_month = (next_month_first - date(year, month, 1)).days
    all_days = [date(year, month, day) for day in range(1, days_in_month + 1)]
    return [d for d in all_days if is_business_day(d, national_calendar)]


def quinto_dia_util(year: int, month: int, national_calendar: holidays.HolidayBase) -> date:
    """Returns the 5th business day of the given year/month."""
    business_days = _business_days_in_month(year, month, national_calendar)
    if len(business_days) < _QUINTO_DIA_UTIL_ORDINAL:
        raise ValueError(
            f"month {year}-{month:02d} has fewer than {_QUINTO_DIA_UTIL_ORDINAL} business days"
        )
    return business_days[_QUINTO_DIA_UTIL_ORDINAL - 1]


def is_quinto_dia_util(d: date, national_calendar: holidays.HolidayBase) -> bool:
    """Returns whether `d` is the 5th business day of its month."""
    return d == quinto_dia_util(d.year, d.month, national_calendar)


def bolsa_familia_window(
    year: int, month: int, national_calendar: holidays.HolidayBase
) -> set[date]:
    """Returns the last ~10 business days of the given year/month (see module docstring)."""
    business_days = _business_days_in_month(year, month, national_calendar)
    return set(business_days[-_BOLSA_FAMILIA_WINDOW_BUSINESS_DAYS:])


def is_bolsa_familia_window(d: date, national_calendar: holidays.HolidayBase) -> bool:
    """Returns whether `d` falls within its month's Bolsa Familia disbursement window."""
    return d in bolsa_familia_window(d.year, d.month, national_calendar)


def _date_only_features(d: date, national_calendar: holidays.HolidayBase) -> dict[str, bool]:
    """Builds the subset of calendar features that depend only on the date, not on estado."""
    return {
        "is_national_holiday": is_national_holiday(d, national_calendar),
        "is_dia_das_maes": is_dia_das_maes(d),
        "is_dia_dos_namorados": is_dia_dos_namorados(d),
        "is_dia_dos_pais": is_dia_dos_pais(d),
        "is_natal": is_natal(d),
        "is_quinto_dia_util": is_quinto_dia_util(d, national_calendar),
        "is_bolsa_familia_window": is_bolsa_familia_window(d, national_calendar),
    }


def _daily_feature_row(
    d: date,
    estado: str,
    national_calendar: holidays.HolidayBase,
    state_calendar: holidays.HolidayBase,
) -> dict[str, object]:
    """Builds one full (date, estado) feature row, combining date-only and state-only features."""
    return {
        "date": pd.Timestamp(d),
        "estado": estado,
        **_date_only_features(d, national_calendar),
        "is_state_holiday": is_state_only_holiday(d, national_calendar, state_calendar),
    }


def _state_calendars_by_estado(
    estados: Sequence[str], years: Sequence[int]
) -> dict[str, holidays.HolidayBase]:
    """Builds one state holiday calendar per estado, keyed by the (already-normalized) UF code."""
    return {estado: _state_holidays(estado, years) for estado in estados}


def _all_daily_feature_rows(
    dates: Sequence[date],
    estados: Sequence[str],
    national_calendar: holidays.HolidayBase,
    state_calendars: dict[str, holidays.HolidayBase],
) -> list[dict[str, object]]:
    """Builds the full (date x estado) cross product of feature rows."""
    return [
        _daily_feature_row(d, estado, national_calendar, state_calendars[estado])
        for d in dates
        for estado in estados
    ]


def build_daily_calendar_features(
    dates: pd.DatetimeIndex | Sequence[date],
    estados: Sequence[str],
) -> pd.DataFrame:
    """Builds one row per (date, estado) with calendar feature columns (see module docstring)."""
    normalized_dates = sorted({_to_date(d) for d in dates})
    normalized_estados = sorted({_validate_estado(e) for e in estados})
    years = sorted({d.year for d in normalized_dates})
    national_calendar = _national_holidays(years)
    state_calendars = _state_calendars_by_estado(normalized_estados, years)

    rows = _all_daily_feature_rows(
        normalized_dates, normalized_estados, national_calendar, state_calendars
    )
    return pd.DataFrame(rows)


def _cycle_date_range(open_date: date, cycle_duration: int) -> list[date]:
    """Returns every calendar date in the cycle window [open_date, open_date + cycle_duration)."""
    return [open_date + timedelta(days=offset) for offset in range(cycle_duration)]


def _cycle_daily_feature_frame(
    estado: str,
    cycle_dates: list[date],
    national_calendar: holidays.HolidayBase,
    state_calendar: holidays.HolidayBase,
) -> pd.DataFrame:
    """Builds the per-day feature frame for a cycle's date window, one row per day."""
    rows = [_daily_feature_row(d, estado, national_calendar, state_calendar) for d in cycle_dates]
    return pd.DataFrame(rows)


def _count_columns(frame: pd.DataFrame) -> dict[str, int]:
    """Sums each boolean count-style column of a cycle's daily feature frame."""
    count_columns = {
        "is_national_holiday": "n_national_holidays",
        "is_state_holiday": "n_state_holidays",
        "is_dia_das_maes": "n_dia_das_maes",
        "is_dia_dos_namorados": "n_dia_dos_namorados",
        "is_dia_dos_pais": "n_dia_dos_pais",
        "is_natal": "n_natal",
        "is_bolsa_familia_window": "n_bolsa_familia_window_days",
    }
    return {out_name: int(frame[col].sum()) for col, out_name in count_columns.items()}


def summarize_cycle_calendar_features(
    estado: str,
    open_date: date | pd.Timestamp,
    cycle_duration: int,
) -> dict[str, int | bool]:
    """Aggregates calendar features over a cycle's day window, for the level dataset's join."""
    normalized_estado = _validate_estado(estado)
    normalized_open_date = _to_date(open_date)
    cycle_dates = _cycle_date_range(normalized_open_date, cycle_duration)
    years = sorted({d.year for d in cycle_dates})
    national_calendar = _national_holidays(years)
    state_calendar = _state_holidays(normalized_estado, years)

    frame = _cycle_daily_feature_frame(
        normalized_estado, cycle_dates, national_calendar, state_calendar
    )
    return {
        **_count_columns(frame),
        "has_quinto_dia_util": bool(frame["is_quinto_dia_util"].any()),
    }
