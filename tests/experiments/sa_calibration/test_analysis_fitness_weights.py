"""Testes da análise com os pesos da fitness (`w_std`, `w_range`) presentes ou ausentes no CSV."""

from __future__ import annotations

import pandas as pd

from src.calibration.analysis import (
    CALIBRATED_PARAMS,
    SEARCH_PARAMS,
    configs_from_runs,
    main_effects,
)

BASE = {
    "initial_temperature": 100.0,
    "min_temperature": 0.1,
    "sector_bias": 0.5,
    "destination_bias": 0.5,
    "penalty_coefficient": 10.0,
}


def _runs(with_weights: bool) -> pd.DataFrame:
    rows = []
    for i in range(8):
        row = {**BASE, "config_id": f"c{i}", "gap": float(i), "cooling_rate": 0.99}
        if with_weights:
            row.update(w_std=0.5 * i, w_range=0.1 * i)
        rows.append(row)
    return pd.DataFrame(rows)


def test_weights_are_search_and_calibrated_params():
    assert {"w_std", "w_range"} <= set(SEARCH_PARAMS) & set(CALIBRATED_PARAMS)


def test_configs_keep_the_weights_when_the_csv_has_them():
    configs = configs_from_runs(_runs(with_weights=True), CALIBRATED_PARAMS)

    assert (configs["c2"]["w_std"], configs["c2"]["w_range"]) == (1.0, 0.2)


def test_csvs_without_weights_still_work():
    runs = _runs(with_weights=False)

    configs = configs_from_runs(runs, CALIBRATED_PARAMS)
    effects = main_effects(runs, SEARCH_PARAMS)

    assert "w_std" not in configs["c0"]
    assert "w_std" not in set(effects["param"])


def test_main_effects_include_the_weights():
    effects = main_effects(_runs(with_weights=True), SEARCH_PARAMS)

    assert {"w_std", "w_range"} <= set(effects["param"])
