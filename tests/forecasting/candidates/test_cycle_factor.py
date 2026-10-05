"""Tests for the level-times-cycle-factor candidate."""

from __future__ import annotations

import pandas as pd

import src.forecasting.candidates  # noqa: F401 - registers candidates into REGISTRY
from src.config.schema import CycleFactorParams
from src.forecasting.model import REGISTRY

CYCLES = ["202601", "202602", "202603", "202604"]
DATES = pd.to_datetime(["2026-01-05", "2026-02-02", "2026-03-02", "2026-04-02"])


def _history(sectors: dict[str, list[float]]) -> pd.DataFrame:
    rows = [
        {"cd_setor": sector, "CICLOS": cycle, "items": value, "opening_date": date}
        for sector, values in sectors.items()
        for cycle, date, value in zip(CYCLES, DATES, values, strict=True)
    ]
    return pd.DataFrame(rows).assign(
        window_start=lambda frame: frame["opening_date"], cycle_days=21
    )


def _targets(sectors: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"cd_setor": s, "CICLOS": "202605", "opening_date": pd.Timestamp("2026-05-04")}
            for s in sectors
        ]
    ).assign(window_start=lambda frame: frame["opening_date"], cycle_days=21)


def test_a_flat_panel_reproduces_the_sector_mean():
    # Every cycle sits exactly on each sector's running mean, so the factor
    # is 1.0 and the candidate must collapse onto naive:mean.
    history = _history({"A": [100, 100, 100, 100], "B": [10, 10, 10, 10]})

    predictions = REGISTRY.build(CycleFactorParams()).fit_predict(history, _targets(["A", "B"]))

    by_sector = predictions.set_index("cd_setor")["items_pred"]
    assert by_sector["A"] == pytest_approx(100)
    assert by_sector["B"] == pytest_approx(10)


def test_a_shared_lift_in_the_last_cycle_is_carried_into_the_forecast():
    # Both sectors run 50% above their own history in the final cycle. That
    # is the shared cycle effect, and "last" must pass it through.
    history = _history({"A": [100, 100, 100, 150], "B": [10, 10, 10, 15]})

    predictions = REGISTRY.build(CycleFactorParams()).fit_predict(history, _targets(["A", "B"]))

    by_sector = predictions.set_index("cd_setor")["items_pred"]
    assert by_sector["A"] > history[history["cd_setor"] == "A"]["items"].mean()
    assert by_sector["B"] > history[history["cd_setor"] == "B"]["items"].mean()


def test_a_sector_without_history_is_skipped_not_guessed():
    history = _history({"A": [100, 100, 100, 100]})

    predictions = REGISTRY.build(CycleFactorParams()).fit_predict(history, _targets(["A", "NEW"]))

    assert set(predictions["cd_setor"]) == {"A"}


def test_the_name_records_how_the_factor_is_forecast():
    assert REGISTRY.build(CycleFactorParams(strategy="mean", window=3)).name == "cycle_factor:mean3"


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, rel=1e-6)
