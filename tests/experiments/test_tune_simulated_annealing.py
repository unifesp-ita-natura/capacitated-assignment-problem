"""Tests for the Optuna SA sweep against a tiny toy instance — no real solve, no forecasting."""

from __future__ import annotations

import optuna

from experiments.tune_simulated_annealing.run import make_objective, run_study, suggest_params

# Same toy instance as tests/solver/heuristics/test_simulated_annealing.py: 2 sectors, 2
# combinations, 2 days, 1 CD, known-optimal amplitude-zero solution.
_DEMAND = {
    (1, 1, 1): 10.0,
    (1, 2, 1): 0.0,
    (1, 1, 2): 0.0,
    (1, 2, 2): 10.0,
    (2, 1, 1): 10.0,
    (2, 2, 1): 0.0,
    (2, 1, 2): 0.0,
    (2, 2, 2): 10.0,
}

_TOY_INSTANCE = dict(
    sectors=[1, 2],
    combinations=[1, 2],
    days=[1, 2],
    cd_sectors={1: [1, 2]},
    daily_capacity={(1, 1): 100.0, (1, 2): 100.0},
    projected_demand=_DEMAND,
    current_assignment={1: 1, 2: 1},
    max_churn=2,
    days_by_cycle={0: [1, 2]},
)

_SWEPT_PARAM_NAMES = {
    "initial_temperature",
    "cooling_rate",
    "min_temperature",
    "stagnation_window",
    "stagnation_tolerance",
    "sector_bias",
    "destination_bias",
}


def test_run_study_runs_the_requested_number_of_trials():
    study = run_study(_TOY_INSTANCE, n_trials=3, seeds=[1], sampler_seed=0)

    assert len(study.trials) == 3
    assert set(study.best_params) == _SWEPT_PARAM_NAMES


_PARAM_BOUNDS = {
    "initial_temperature": (1e2, 1e5),
    "cooling_rate": (0.90, 0.999),
    "min_temperature": (1e-3, 10.0),
    "stagnation_window": (50, 500),
    "stagnation_tolerance": (1e-6, 1e-2),
    "sector_bias": (0.0, 1.0),
    "destination_bias": (0.0, 1.0),
}


def test_suggest_params_stays_within_configured_bounds():
    study = optuna.create_study()
    trial = study.ask()

    params = suggest_params(trial)

    for field, (low, high) in _PARAM_BOUNDS.items():
        assert low <= getattr(params, field) <= high, field


def test_objective_records_diagnostics_as_user_attrs():
    study = optuna.create_study()
    objective = make_objective(_TOY_INSTANCE, seeds=[1])

    study.optimize(objective, n_trials=1)

    attrs = study.trials[0].user_attrs
    assert {"objective", "capacity_penalty", "churn_penalty"} <= set(attrs)
