"""Tests for scenario queries: f(sector, window start, cycle length) through the harness."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.dataset import build_item_panel, load_demand_base
from src.forecasting.model import align_predictions, forecast, per_sector

HISTORY = pd.DataFrame(
    {
        "cd_setor": ["A", "A", "B", "B"],
        "CICLOS": ["1", "2", "1", "2"],
        "items": [100, 200, 10, 20],
        "opening_date": pd.to_datetime(["2026-01-01", "2026-01-22"] * 2),
        "window_start": pd.to_datetime(["2026-01-01", "2026-01-22"] * 2),
        "cycle_days": 21,
    }
)
# Three opening days for A in the next cycle, one of them never seen before.
QUERIES = pd.DataFrame(
    {
        "cd_setor": ["A", "A", "A", "B"],
        "window_start": pd.to_datetime(["2026-02-12", "2026-02-13", "2026-02-14", "2026-02-12"]),
        "cycle_days": [21, 21, 14, 21],
    }
)
LAST_VALUE = per_sector(
    name="last", fn=lambda series, horizon: pd.Series([series.iloc[-1]] * horizon)
)


class _PerDayCandidate:
    """Scenario-aware: the last cycle's items per day, times the queried window length."""

    name = "per_day"

    def fit_predict(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        last = history.sort_values("opening_date").groupby("cd_setor").last()
        rate = last["items"] / last["cycle_days"]
        return targets.assign(items_pred=targets["cycle_days"] * targets["cd_setor"].map(rate))


def test_a_window_blind_candidate_gives_every_scenario_of_a_cycle_the_same_value():
    answered = forecast(LAST_VALUE, HISTORY, QUERIES)

    assert answered["items_pred"].tolist() == [200, 200, 200, 20]


def test_a_scenario_aware_candidate_answers_each_query_on_its_own():
    answered = forecast(_PerDayCandidate(), HISTORY, QUERIES)

    # A sold 200 items in 21 days: a 14-day window gets two thirds of that.
    assert answered["items_pred"].tolist() == pytest.approx([200, 200, 200 * 14 / 21, 20])


def test_forecast_keeps_the_query_order_and_marks_a_skipped_sector_as_nan():
    answered = forecast(LAST_VALUE, HISTORY[HISTORY["cd_setor"] == "A"], QUERIES)

    assert answered["cd_setor"].tolist() == QUERIES["cd_setor"].tolist()
    assert answered["items_pred"].isna().tolist() == [False, False, False, True]


def test_forecast_refuses_a_query_inside_the_history():
    past = QUERIES.assign(window_start=pd.Timestamp("2026-01-22"))

    with pytest.raises(ValueError, match="after the end"):
        forecast(LAST_VALUE, HISTORY, past)


def test_align_predictions_matches_the_full_query_when_the_candidate_returns_it():
    targets = QUERIES.assign(CICLOS="3")
    predictions = targets.iloc[::-1].assign(items_pred=[4.0, 3.0, 2.0, 1.0])

    assert align_predictions(targets, predictions).tolist() == [1.0, 2.0, 3.0, 4.0]


def test_the_loader_keeps_one_copy_of_an_order_repeated_under_two_opening_dates(tmp_path):
    raw = pd.read_csv("tests/fixtures/demand_sample.csv").drop(columns="Qtde dias")
    copy = raw.iloc[[0]].assign(**{"Dt Abertura": "2025-12-01", "dia_ciclo": 5})
    path = tmp_path / "fanned_out.csv"
    pd.concat([raw, copy]).to_csv(path, index=False)

    loaded = load_demand_base(path)

    assert len(loaded) == len(raw)
    assert loaded["total_itens_mascarado"].sum() == raw["total_itens_mascarado"].sum()
    # The copy under the earlier opening date is the one dropped.
    assert loaded.iloc[0]["Dt Abertura"] == pd.Timestamp(raw.iloc[0]["Dt Abertura"])
    # `Qtde dias`, missing from base_tratada_v2, is rebuilt from the window.
    assert loaded.iloc[0]["Qtde dias"] == 25


def test_the_panel_carries_each_sectors_own_window():
    panel = build_item_panel(load_demand_base("tests/fixtures/demand_sample.csv"))

    assert (panel["window_start"] >= panel["opening_date"]).all()
    assert panel["cycle_days"].between(1, 40).all()
