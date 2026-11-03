"""Testes de como a calibração repassa e registra a fitness (`energy_mode` e pesos) do SA."""

from __future__ import annotations

from src.calibration.config import BudgetSettings
from src.calibration.runner import RunSpec, resolve_params, run_record, run_sa

CONFIG = {
    "initial_temperature": 100.0,
    "min_temperature": 0.1,
    "sector_bias": 0.5,
    "destination_bias": 0.5,
    "penalty_coefficient": 1000.0,
}


def test_default_budget_keeps_the_range_energy(small_budget):
    assert resolve_params(CONFIG, small_budget).energy_mode == "range"


def test_budget_energy_mode_reaches_every_run():
    budget = BudgetSettings(max_iterations=500, energy_mode="peak_std_breach")

    params = resolve_params({**CONFIG, "w_std": 2.0, "w_range": 0.5}, budget)

    assert params.energy_mode == "peak_std_breach"
    assert (params.w_peak, params.w_std, params.w_range) == (1.0, 2.0, 0.5)


def test_config_energy_mode_overrides_the_budget(small_budget):
    params = resolve_params({**CONFIG, "energy_mode": "peak_std_breach"}, small_budget)

    assert params.energy_mode == "peak_std_breach"


def test_run_record_reports_the_fitness(tiny_instance):
    budget = BudgetSettings(max_iterations=500, trace_every=100, energy_mode="peak_std_breach")
    run = run_sa(
        tiny_instance, RunSpec("screen", "c0", {**CONFIG, "w_std": 3.0}, "tiny", 1), budget
    )

    record = run_record(run)

    assert record["energy_mode"] == "peak_std_breach"
    assert (record["w_peak"], record["w_std"], record["w_range"]) == (1.0, 3.0, 0.0)
    assert record["objective"] == 0.0
