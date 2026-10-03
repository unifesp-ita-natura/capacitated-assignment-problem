"""Testes do executor da calibração: parâmetros efetivos, traço, registro e expansão."""

from __future__ import annotations

import pytest

from src.calibration.runner import (
    RunSpec,
    TraceRecorder,
    expand_specs,
    resolve_params,
    run_record,
    run_sa,
)
from src.solver.heuristics.simulated_annealing import cooling_rate_for_budget

CONFIG = {
    "initial_temperature": 100.0,
    "min_temperature": 0.1,
    "sector_bias": 0.5,
    "destination_bias": 0.5,
    "penalty_coefficient": 1000.0,
}


def test_resolve_params_derives_alpha_and_disables_stagnation(small_budget):
    params = resolve_params(CONFIG, small_budget)

    assert params.cooling_rate == pytest.approx(cooling_rate_for_budget(500, 100.0, 0.1))
    assert params.max_iterations == params.stagnation_window == 500
    assert params.stagnation_tolerance == 0.0


def test_resolve_params_keeps_an_explicit_alpha(small_budget):
    params = resolve_params({**CONFIG, "cooling_rate": 0.9}, small_budget)

    assert params.cooling_rate == 0.9


def test_trace_recorder_keeps_every_nth_iteration_and_the_running_best():
    recorder = TraceRecorder(every=2)
    for iteration, energy in enumerate([5.0, 3.0, 4.0, 1.0, 2.0]):
        recorder(iteration=iteration, energy=energy, temperature=1.0, assignment=None)

    assert recorder.rows == [(0, 5.0, 5.0, 1.0), (2, 4.0, 3.0, 1.0), (4, 2.0, 1.0, 1.0)]


def test_run_sa_on_the_two_sector_instance(tiny_instance, small_budget):
    spec = RunSpec("screen", "c0", CONFIG, "tiny", seed=1)

    run = run_sa(tiny_instance, spec, small_budget)

    assert run.result.objective == 0.0
    assert run.result.feasible
    assert abs(run.result.iterations - 500) <= 1
    assert len(run.trace) == 5  # iterações 0, 100, ..., 400


def test_run_record_uses_the_report_metric_names(tiny_instance, small_budget):
    run = run_sa(tiny_instance, RunSpec("screen", "c0", CONFIG, "tiny", seed=1), small_budget)

    record = run_record(run)

    assert {"p_cap", "p_churn", "zmax", "zmin", "std_load", "feasible", "time"} <= set(record)
    assert (record["phase"], record["config_id"], record["instance"], record["seed"]) == (
        "screen",
        "c0",
        "tiny",
        1,
    )
    assert record["scheduled_iterations"] <= 500


def test_expand_specs_is_the_full_product():
    specs = expand_specs("tpe", {"a": CONFIG, "b": CONFIG}, ["x", "y"], [1, 2, 3])

    assert len(specs) == 12
    assert len({spec.key for spec in specs}) == 12
    assert specs[0].key == ("tpe", "a", "x", 1)
