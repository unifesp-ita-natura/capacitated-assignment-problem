"""Testes da comparação de solvers: referência do gap (ótimo provado) e agregação das tabelas."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.calibration.comparison import (
    best_lower_bound,
    group_by_instance,
    load_comparison_config,
    mip_record_name,
    mip_rows,
    proven_optimum,
    reference_upper_bound,
    sa_rows,
    solve_options,
)

CONFIG_PATH = Path("configs/experiments/sa_calibration/solver_comparison.yaml")


def _record(solver, gap, ub, lb=None, termination="optimal", instance="i"):
    return {
        "instance": instance,
        "solver": solver,
        "mip_rel_gap": gap,
        "termination": termination,
        "upper_bound": ub,
        "lower_bound": lb if lb is not None else ub,
        "gap": 0.0,
        "time": 1.0,
    }


def test_proven_optimum_is_the_best_closed_ub_at_the_tight_tolerance():
    records = [
        _record("appsi_gurobi", 1e-4, 100.005),
        _record("appsi_highs", 1e-4, 100.0),
        _record("appsi_gurobi", 0.01, 99.0),  # tolerância frouxa não entra
    ]

    assert proven_optimum(records, 1e-4) == 100.0


def test_proven_optimum_ignores_runs_that_did_not_close():
    records = [_record("appsi_highs", 1e-4, 90.0, termination="maxTimeLimit")]

    assert proven_optimum(records, 1e-4) is None


def test_disagreeing_proven_optima_stop_the_comparison():
    records = [_record("appsi_gurobi", 1e-4, 100.0), _record("appsi_highs", 1e-4, 101.0)]

    with pytest.raises(ValueError, match="discordam"):
        proven_optimum(records, 1e-4)


def test_reference_upper_bound_picks_the_solver_and_tolerance():
    records = [_record("appsi_highs", 0.01, 102.0), _record("appsi_gurobi", 0.01, 101.0)]

    assert reference_upper_bound(records, "appsi_highs", 0.01) == 102.0
    assert reference_upper_bound(records, "appsi_highs", 1e-4) is None


def test_mip_rows_report_the_gap_of_each_solution_against_z_ref():
    rows = mip_rows([_record("appsi_gurobi", 0.01, 102.0, lb=100.0)], {"i": 100.0})

    assert rows.loc[0, "method"] == "appsi_gurobi (gap 0.01)"
    assert rows.loc[0, "gap_vs_zref_pct"] == pytest.approx(2.0)


def test_sa_rows_aggregate_seeds_and_count_infeasible_as_100():
    runs = pd.DataFrame(
        {
            "instance": ["i", "i"],
            "config_id": ["c", "c"],
            "objective": [110.0, 120.0],
            "feasible": [True, False],
            "zmax": [500.0, 520.0],
            "std_load": [30.0, 32.0],
            "time": [1.0, 3.0],
        }
    )

    row = sa_rows(runs, {"i": 100.0}, {"loose": {"i": 110.0}}).iloc[0]

    assert row["status"] == "1/2 viáveis"
    assert row["gap_vs_zref_pct"] == pytest.approx((10.0 + 100.0) / 2)
    assert row["gap_vs_loose_pct"] == pytest.approx((0.0 + 100.0) / 2)
    assert (row["peak"], row["std_load"], row["time_s"]) == (510.0, 31.0, 2.0)


def test_sa_rows_without_reference_give_nan_gap():
    runs = pd.DataFrame(
        {
            "instance": ["i"],
            "config_id": ["c"],
            "objective": [1.0],
            "feasible": [True],
            "zmax": [1.0],
            "std_load": [0.0],
            "time": [1.0],
        }
    )

    assert math.isnan(sa_rows(runs, {"i": None}, {"x": {"i": None}}).iloc[0]["gap_vs_zref_pct"])


def test_best_lower_bound_is_the_largest_available():
    records = [_record("a", 1e-4, 10.0, lb=8.0), _record("b", 1e-4, 10.0, lb=9.0)]

    assert best_lower_bound(records) == 9.0
    assert best_lower_bound([{"lower_bound": None}]) is None


def test_mip_record_name():
    assert mip_record_name("base40", "appsi_highs", 1e-4) == "base40_appsi_highs_gap0.0001"


def test_the_repository_yaml_is_valid():
    config = load_comparison_config(CONFIG_PATH)

    assert (config.reference_gap, config.loose_gap) == (1e-4, 0.01)
    assert {c["energy_mode"] for c in config.sa_configs.values()} == {"range", "peak_std_breach"}


def test_solve_options_fill_placeholders_and_create_dirs(tmp_path):
    config = load_comparison_config(CONFIG_PATH).model_copy(
        update={
            "mip_extra_options": {
                "appsi_gurobi": {
                    "Threads": 2,
                    "NodefileDir": str(tmp_path / "nodes" / "{instance}"),
                    "LogFile": str(tmp_path / "logs" / "{instance}_{gap}.log"),
                }
            }
        }
    )

    options = solve_options(config, "appsi_gurobi", "s800", 1e-4)

    assert options["Threads"] == 2
    assert options["LogFile"] == str(tmp_path / "logs" / "s800_0.0001.log")
    assert (tmp_path / "nodes" / "s800").is_dir() and (tmp_path / "logs").is_dir()
    assert solve_options(config, "appsi_highs", "s800", 1e-4) == {}


def test_group_by_instance_keeps_instances_without_records():
    grouped = group_by_instance([_record("a", 1e-4, 1.0, instance="x")], ["x", "y"])

    assert len(grouped["x"]) == 1 and grouped["y"] == []
