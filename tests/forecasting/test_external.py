"""Tests for the per-UF external features and their attachment to the item panel."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.dataset import build_item_panel, load_demand_base
from src.forecasting.external import EXTERNAL_COLUMNS, load_uf_features

FIXTURE_PATH = "tests/fixtures/demand_sample.csv"


def test_load_uf_features_gives_one_complete_row_per_state():
    dates = pd.Series(pd.to_datetime(["2025-01-10", "2025-12-20"]))

    features = load_uf_features(dates)

    assert list(features.columns) == ["estado", *EXTERNAL_COLUMNS]
    assert features["estado"].is_unique
    assert len(features) == 27
    assert features[EXTERNAL_COLUMNS].notna().all().all()


def test_build_item_panel_attaches_external_features_by_state():
    raw = load_demand_base(FIXTURE_PATH)

    panel = build_item_panel(raw, external=True)

    assert set(EXTERNAL_COLUMNS) <= set(panel.columns)
    assert panel[EXTERNAL_COLUMNS].notna().all().all()
    # static per state: every row of one state carries the same values
    assert (panel.groupby("estado")[EXTERNAL_COLUMNS].nunique() == 1).all().all()


def test_build_item_panel_has_no_external_columns_by_default():
    panel = build_item_panel(load_demand_base(FIXTURE_PATH))

    assert not set(EXTERNAL_COLUMNS) & set(panel.columns)


def test_external_features_need_the_by_city_panel():
    raw = load_demand_base(FIXTURE_PATH)

    with pytest.raises(ValueError, match="estado"):
        build_item_panel(raw, by_city=False, external=True)
