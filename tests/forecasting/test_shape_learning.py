"""Verify organization/calendar predictors and historical-only normalized shape fitting."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.shape_learning import (
    ShapeLearningParams,
    feature_table,
    historical_windows,
    predict_learned_shape,
)


@pytest.fixture
def history():
    return pd.DataFrame(
        {
            "cd_setor": ["s"] * 4,
            "CICLOS": ["1"] * 4,
            "date": pd.date_range("2026-01-01", periods=4),
            "window_start": pd.Timestamp("2026-01-01"),
            "n_days": 4,
            "offset": range(4),
            "CD_RE": "r",
            "CD_GV": "g",
            "actual_share": [0.1, 0.2, 0.3, 0.4],
            "total_actual": 100,
            "opening_date": pd.Timestamp("2026-01-01"),
        }
    )


def test_organization_and_calendar_features(history):
    features = feature_table(history, ["CD_RE", "CD_GV", "weekday", "month"])
    assert features["CD_RE"].tolist() == ["r"] * 4
    assert features["weekday"].tolist() == [3, 4, 5, 6]
    assert features["month"].tolist() == [1] * 4


def test_feature_selection(history):
    assert feature_table(history, []).columns.tolist() == [
        "cd_setor",
        "relative_position",
        "n_days",
    ]


def test_future_assignments_are_not_read(history):
    actual = history.assign(CICLOS="2", CD_RE="future_region", CD_GV="future_manager")
    windows = historical_windows(history, actual)
    assert windows.iloc[0]["CD_RE"] == "r"
    assert windows.iloc[0]["CD_GV"] == "g"


def test_learned_shape_conserves_total(history):
    windows = historical_windows(history, history.assign(CICLOS="2"))
    params = ShapeLearningParams(n_estimators=5, min_child_samples=1)
    forecast = predict_learned_shape(history, windows, params)
    assert forecast["share_pred"].sum() == pytest.approx(1)
    assert forecast["share_pred"].ge(0).all()


def test_unseen_categories_are_supported(history):
    windows = historical_windows(history, history.assign(CICLOS="2"))
    windows["CD_RE"] = "new"
    forecast = predict_learned_shape(history, windows, ShapeLearningParams(n_estimators=2))
    assert forecast["share_pred"].sum() == pytest.approx(1)


def test_zero_history_falls_back_to_uniform(history):
    windows = historical_windows(history, history.assign(CICLOS="2"))
    forecast = predict_learned_shape(history.assign(total_actual=0), windows, ShapeLearningParams())
    np.testing.assert_allclose(forecast["share_pred"], 0.25)
    assert forecast["uniform_fallback"].all()


def test_invalid_feature_rejected():
    with pytest.raises(ValueError):
        ShapeLearningParams(features=["unsupported"])


def test_holidays_not_available():
    with pytest.raises(ValueError):
        ShapeLearningParams(features=["is_holiday"])
