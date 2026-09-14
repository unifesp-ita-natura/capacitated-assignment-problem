"""Tests for the Strategy interface, the per-sector Adapter, and the Registry/Factory."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import BaseModel

from src.forecasting.model import CandidateRegistry, InsufficientHistoryError, per_sector

HISTORY = pd.DataFrame(
    {
        "cd_setor": ["A", "A", "B", "B"],
        "CICLOS": ["1", "2", "1", "2"],
        "items": [100, 200, 10, 20],
        "opening_date": pd.to_datetime(["2026-01-01", "2026-01-22", "2026-01-01", "2026-01-22"]),
    }
)
TARGETS = pd.DataFrame(
    {
        "cd_setor": ["A", "B"],
        "CICLOS": ["3", "3"],
        "opening_date": pd.to_datetime(["2026-02-12", "2026-02-12"]),
    }
)


def test_per_sector_applies_fn_independently_to_each_sector():
    candidate = per_sector(
        name="last", fn=lambda series, horizon: pd.Series([series.iloc[-1]] * horizon)
    )

    predictions = candidate.fit_predict(HISTORY, TARGETS)

    by_sector = predictions.set_index("cd_setor")["items_pred"]
    assert by_sector["A"] == 200
    assert by_sector["B"] == 20


def test_per_sector_skips_a_sector_with_no_target_cycles_without_calling_fn():
    # A sector can have history but sit out the target cycle entirely (it
    # simply didn't order that cycle) — the Adapter must skip it rather than
    # call `fn` for a prediction nothing will be scored against.
    calls: list[str] = []

    def fn(series: pd.Series, horizon: int) -> pd.Series:
        calls.append("called")
        return pd.Series([series.iloc[-1]] * horizon)

    targets_missing_b = TARGETS[TARGETS["cd_setor"] == "A"]
    candidate = per_sector(name="last", fn=fn)

    predictions = candidate.fit_predict(HISTORY, targets_missing_b)

    assert set(predictions["cd_setor"]) == {"A"}
    assert calls == ["called"]  # fn ran once, for A only — never for B


def test_per_sector_skips_a_sector_whose_fn_raises_insufficient_history():
    def fn(series: pd.Series, horizon: int) -> pd.Series:
        if series.iloc[-1] == 20:  # sector B
            raise InsufficientHistoryError("not enough data")
        return pd.Series([series.iloc[-1]] * horizon)

    candidate = per_sector(name="picky", fn=fn)

    predictions = candidate.fit_predict(HISTORY, TARGETS)

    assert set(predictions["cd_setor"]) == {"A"}


def test_per_sector_propagates_unexpected_errors():
    def fn(series: pd.Series, horizon: int) -> pd.Series:
        raise RuntimeError("real bug")

    candidate = per_sector(name="broken", fn=fn)

    with pytest.raises(RuntimeError, match="real bug"):
        candidate.fit_predict(HISTORY, TARGETS)


class _DummyParams(BaseModel):
    model: str = "dummy"


def test_registry_round_trips_a_registered_builder():
    registry = CandidateRegistry()

    @registry.register("dummy")
    def _build(params: _DummyParams):
        return per_sector(name="dummy", fn=lambda s, h: pd.Series([0] * h))

    candidate = registry.build(_DummyParams())

    assert candidate.name == "dummy"


def test_registry_raises_a_helpful_error_for_an_unknown_model():
    registry = CandidateRegistry()

    with pytest.raises(ValueError, match="no candidate registered"):
        registry.build(_DummyParams())


def test_registry_known_models_lists_registered_names():
    registry = CandidateRegistry()
    registry.register("dummy")(
        lambda params: per_sector(name="dummy", fn=lambda s, h: pd.Series([0] * h))
    )

    assert registry.known_models == ["dummy"]
