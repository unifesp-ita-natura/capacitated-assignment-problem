"""Tests for scenario queries: every candidate answers f(sector, cycle, opening date)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.forecasting.candidates  # noqa: F401 - registers candidates into REGISTRY
from src.config.schema import (
    ArimaParams,
    CycleFactorParams,
    ETSParams,
    LightGBMParams,
    NaiveParams,
)
from src.forecasting.dataset import build_item_panel, load_demand_base
from src.forecasting.model import (
    REGISTRY,
    align_predictions,
    cycle_days_exponent,
    forecast,
    opening_scenarios,
    per_sector,
)

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
        "CICLOS": "3",
        "opening_date": pd.Timestamp("2026-02-12"),
    }
)
ALL_CANDIDATES = [
    NaiveParams(strategy="last_value"),
    NaiveParams(strategy="mean"),
    ArimaParams(order=(1, 0, 0), trend="c"),
    ETSParams(),
    ETSParams(alpha="pooled"),
    CycleFactorParams(),
    LightGBMParams(lags=[1], rolling_windows=[2], n_estimators=50, num_leaves=3),
]
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


def test_a_history_that_never_varied_the_length_gives_every_scenario_the_same_value():
    # Every past window lasted 21 days, so nothing says how length matters.
    answered = forecast(LAST_VALUE, HISTORY, QUERIES)

    assert answered["items_pred"].tolist() == pytest.approx([200, 200, 200, 20])


def test_per_sector_scales_each_scenario_to_its_window_length():
    # A sells 10 items a day whatever the window: the exponent comes out 1,
    # so the last value is read as 10 a day and a 14-day window gets 140.
    days = [21, 14, 21]
    history = pd.DataFrame(
        {
            "cd_setor": "A",
            "CICLOS": ["1", "2", "3"],
            "items": [10.0 * d for d in days],
            "opening_date": pd.to_datetime(["2026-01-01", "2026-01-22", "2026-02-05"]),
            "window_start": pd.to_datetime(["2026-01-01", "2026-01-22", "2026-02-05"]),
            "cycle_days": days,
        }
    )

    answered = forecast(LAST_VALUE, history, QUERIES[QUERIES["cd_setor"] == "A"])

    assert answered["items_pred"].tolist() == pytest.approx([210, 210, 140])


def test_cycle_days_exponent_reads_how_items_follow_the_length():
    days = [21, 14, 21, 14]
    panel = pd.DataFrame({"cd_setor": ["A"] * 4 + ["B"] * 4, "cycle_days": days * 2})

    per_day = panel.assign(items=[10.0 * d for d in days] + [d for d in days])
    flat = panel.assign(items=[200.0] * 4 + [20.0] * 4)

    assert cycle_days_exponent(per_day) == pytest.approx(1.0)
    assert cycle_days_exponent(flat) == 0.0


@pytest.mark.parametrize("params", ALL_CANDIDATES, ids=lambda params: params.model)
def test_every_candidate_scales_to_the_window_length(params):
    # 20 sectors selling 10 items a day over windows of random length. Asked
    # about the same cycle opening on two dates that leave 21 and 14 days,
    # every model must give the shorter window less.
    lengths = np.random.default_rng(0).choice([14, 21], size=(20, 10))
    starts = pd.Timestamp("2025-01-06") + pd.to_timedelta(21 * np.arange(10), unit="D")
    history = pd.DataFrame(
        [
            {
                "cd_setor": f"S{s}",
                "CICLOS": f"2025{n:02d}",
                "items": 10.0 * lengths[s, n] * (1 + s),
                "opening_date": starts[n],
                "window_start": starts[n],
                "cycle_days": int(lengths[s, n]),
            }
            for s in range(20)
            for n in range(10)
        ]
    )
    opening = pd.Timestamp("2025-07-21")
    queries = pd.DataFrame(
        {
            "cd_setor": "S0",
            "CICLOS": "202510",
            "opening_date": opening,
            "window_start": [opening, opening + pd.Timedelta(days=7)],
            "cycle_days": [21, 14],
        }
    )

    answered = forecast(REGISTRY.build(params), history, queries)

    long_window, short_window = answered["items_pred"]
    assert short_window < long_window


def test_opening_scenarios_move_the_window_and_keep_its_length():
    cycles = pd.DataFrame(
        {
            "cd_setor": ["A", "B"],
            "CICLOS": "3",
            "opening_date": pd.Timestamp("2026-02-12"),
            "cycle_days": [21, 14],
        }
    )

    scenarios = opening_scenarios(cycles, range(3))

    a = scenarios[scenarios["cd_setor"] == "A"]
    assert a["window_start"].tolist() == list(pd.date_range("2026-02-12", periods=3))
    assert a["cycle_days"].tolist() == [21, 21, 21]
    assert len(scenarios) == 6


@pytest.mark.parametrize("params", ALL_CANDIDATES, ids=lambda params: params.model)
def test_every_candidate_answers_the_same_cycle_opening_on_different_days(params):
    # Cycles open on the 1st; each sector opens on day 0 or day 15 of a
    # cycle at random, always for 21 days, and sells 50% more on day 15.
    # The same cycle, opened on day 15 instead of day 0, must forecast more.
    opens_late = np.random.default_rng(1).random((20, 10)) < 0.5
    openings = pd.date_range("2025-01-01", periods=11, freq="MS")
    history = pd.DataFrame(
        [
            {
                "cd_setor": f"S{s}",
                "CICLOS": f"2025{n:02d}",
                "items": 100.0 * (1 + s) * (1.5 if opens_late[s, n] else 1.0),
                "opening_date": openings[n],
                "window_start": openings[n] + pd.Timedelta(days=15 * opens_late[s, n]),
                "cycle_days": 21,
            }
            for s in range(20)
            for n in range(10)
        ]
    )
    cycle = pd.DataFrame(
        {"cd_setor": ["S0"], "CICLOS": "202510", "opening_date": openings[10], "cycle_days": 21}
    )

    answered = forecast(REGISTRY.build(params), history, opening_scenarios(cycle, [0, 15]))

    day_0, day_15 = answered["items_pred"]
    assert day_15 > day_0


def test_align_predictions_refuses_predictions_without_their_scenario():
    predictions = QUERIES[["cd_setor", "CICLOS"]].assign(items_pred=1.0)

    with pytest.raises(ValueError, match="window_scaled"):
        align_predictions(QUERIES, predictions)


def test_a_scenario_aware_candidate_answers_each_query_on_its_own():
    answered = forecast(_PerDayCandidate(), HISTORY, QUERIES)

    # A sold 200 items in 21 days: a 14-day window gets two thirds of that.
    assert answered["items_pred"].tolist() == pytest.approx([200, 200, 200 * 14 / 21, 20])


def test_forecast_keeps_the_query_order_and_marks_a_skipped_sector_as_nan():
    answered = forecast(LAST_VALUE, HISTORY[HISTORY["cd_setor"] == "A"], QUERIES)

    assert answered["cd_setor"].tolist() == QUERIES["cd_setor"].tolist()
    assert answered["items_pred"].isna().tolist() == [False, False, False, True]


def test_forecast_refuses_a_query_without_its_cycle():
    with pytest.raises(ValueError, match="missing"):
        forecast(LAST_VALUE, HISTORY, QUERIES.drop(columns=["CICLOS", "opening_date"]))


def test_forecast_refuses_a_query_inside_the_history():
    past = QUERIES.assign(opening_date=pd.Timestamp("2026-01-22"))

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
