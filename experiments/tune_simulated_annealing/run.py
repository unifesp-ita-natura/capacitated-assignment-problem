"""hyperparameter sweep over SA search-dynamics hyperparameters against a fixed instance."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from statistics import fmean

import numpy as np
import optuna

from src.forecasting.data import forecast_items, items_per_order_by_sector
from src.forecasting.model import forecast_future_cycles
from src.forecasting.scoring import score_and_select_strategies
from src.generate import generator
from src.solver import inputs as solver_inputs
from src.solver.heuristics.simulated_annealing import AnnealingParams, solve

# Swept in `suggest_params`; kept out of AnnealingParams' own defaults so a
# study run never mutates production behavior on its own.
N_TRIALS = 50
SEEDS = [1, 2, 3]
SAMPLER_SEED = 42
SWEEP_MAX_ITERATIONS = 100_000
OUTPUT_PATH = "experiments/tune_simulated_annealing/outputs/trials.csv"

# Mirrors src/main.py's own defaults, so the tuned instance matches the one
# the pipeline actually solves.
DEFAULT_N_SECTORS = 80
DEFAULT_CYCLE_LENGTH = 21
DEFAULT_N_CYCLES_HISTORY = 6
DEFAULT_N_CYCLES_HORIZON = 4
DEFAULT_MAX_CHURN = 0.2
DEFAULT_CAPACITY_MULTIPLIER = 1.0
DEFAULT_INSTANCE_SEED = 42


def suggest_params(trial: optuna.Trial) -> AnnealingParams:
    """One `AnnealingParams` candidate per trial, covering the search-dynamics fields."""
    return AnnealingParams(
        initial_temperature=trial.suggest_float("initial_temperature", 1e2, 1e5, log=True),
        cooling_rate=trial.suggest_float("cooling_rate", 0.90, 0.999),
        min_temperature=trial.suggest_float("min_temperature", 1e-3, 10.0, log=True),
        max_iterations=SWEEP_MAX_ITERATIONS,
        stagnation_window=trial.suggest_int("stagnation_window", 50, 500),
        stagnation_tolerance=trial.suggest_float("stagnation_tolerance", 1e-6, 1e-2, log=True),
        sector_bias=trial.suggest_float("sector_bias", 0.0, 1.0),
        destination_bias=trial.suggest_float("destination_bias", 0.0, 1.0),
    )


_DIAGNOSTIC_FIELDS = ("objective", "capacity_penalty", "churn_penalty")


def _mean_result(
    instance: dict, params: AnnealingParams, seeds: list[int]
) -> tuple[float, dict[str, float]]:
    """Mean energy (the Optuna objective) and mean diagnostics, across `seeds`."""
    results = [solve(**instance, params=params, rng=random.Random(seed)) for seed in seeds]
    mean_energy = fmean(r.energy for r in results)
    diagnostics = {field: fmean(getattr(r, field) for r in results) for field in _DIAGNOSTIC_FIELDS}
    return mean_energy, diagnostics


def make_objective(instance: dict, seeds: list[int]) -> Callable[[optuna.Trial], float]:
    """Build the Optuna objective closing over the fixed `instance` and evaluation `seeds`."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        mean_energy, diagnostics = _mean_result(instance, params, seeds)
        for name, value in diagnostics.items():
            trial.set_user_attr(name, value)
        return mean_energy

    return objective


def run_study(instance: dict, n_trials: int, seeds: list[int], sampler_seed: int) -> optuna.Study:
    """Run the sweep and return the finished study."""
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=sampler_seed)
    )
    study.optimize(make_objective(instance, seeds), n_trials=n_trials)
    return study


def build_tuning_instance(
    n_sectors: int = DEFAULT_N_SECTORS,
    cycle_length: int = DEFAULT_CYCLE_LENGTH,
    n_cycles_history: int = DEFAULT_N_CYCLES_HISTORY,
    n_cycles_horizon: int = DEFAULT_N_CYCLES_HORIZON,
    max_churn: float = DEFAULT_MAX_CHURN,
    capacity_multiplier: float = DEFAULT_CAPACITY_MULTIPLIER,
    seed: int = DEFAULT_INSTANCE_SEED,
) -> dict:
    """Assemble one SA-ready instance via the same generate -> forecast -> solver_inputs
    pipeline `src/main.py` uses, so the sweep tunes against a representative problem."""
    rng = np.random.default_rng(seed)
    sectors = generator.build_sectors(n_sectors)

    demand_level, demand_shape, cycle_starts, assignment = generator.generate_synthetic_demand(
        rng, sectors, cycle_length, n_cycles_history
    )
    best_level_name, best_shape_name = score_and_select_strategies(
        demand_level, demand_shape, cycle_starts, assignment, n_cycles_history, n_cycles_horizon
    )
    calendar_forecast = forecast_future_cycles(
        demand_level,
        demand_shape,
        cycle_starts,
        assignment,
        n_cycles_history,
        cycle_length,
        best_level_name,
        best_shape_name,
        n_cycles_horizon,
    )

    sector_ids, combo_slots = solver_inputs.build_id_maps(sectors)
    combo_ids = {slot: combo_id for combo_id, slot in combo_slots.items()}

    items_per_order = items_per_order_by_sector(demand_level)
    calendar_forecast = forecast_items(calendar_forecast, items_per_order)

    cycle_span = generator.cycle_span_business_days(cycle_length)
    projected_demand = solver_inputs.build_projected_demand(
        calendar_forecast, sector_ids, combo_slots, cycle_span, n_cycles_history + 1
    )
    days = sorted({day for _, day, _ in projected_demand})
    days_by_cycle = solver_inputs.group_days_by_cycle(days, cycle_span)

    sector_to_cd = generator.build_sector_cd_assignment(rng, sectors)
    cd_sectors = solver_inputs.build_cd_sectors(sector_to_cd, sector_ids)
    daily_capacity = solver_inputs.build_daily_capacity(list(cd_sectors), days, capacity_multiplier)

    current_assignment_sa = solver_inputs.build_current_assignment_sa(
        assignment, sector_ids, combo_ids
    )
    max_churn_sectors = max_churn * n_sectors

    return {
        "sectors": list(sector_ids.values()),
        "combinations": list(combo_slots.keys()),
        "days": days,
        "cd_sectors": cd_sectors,
        "daily_capacity": daily_capacity,
        "projected_demand": projected_demand,
        "current_assignment": current_assignment_sa,
        "max_churn": max_churn_sectors,
        "days_by_cycle": days_by_cycle,
    }


def run(n_trials: int = N_TRIALS) -> optuna.Study:
    """Build the tuning instance once and run the sweep against it."""
    instance = build_tuning_instance()
    return run_study(instance, n_trials, SEEDS, SAMPLER_SEED)


if __name__ == "__main__":
    result_study = run()
    print(f"Best value (mean energy): {result_study.best_value}")
    print(f"Best params: {result_study.best_params}")
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_study.trials_dataframe().to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote trial table to {OUTPUT_PATH}")
