"""Minimal Pyomo model solvable by any Pyomo-registered exact MIP backend."""

from __future__ import annotations

import pyomo.environ as pyo


def build_example_model() -> pyo.ConcreteModel:
    model = pyo.ConcreteModel()
    model.x = pyo.Var(within=pyo.NonNegativeReals)
    model.y = pyo.Var(within=pyo.NonNegativeReals)
    model.capacity = pyo.Constraint(expr=model.x + model.y <= 10)
    model.objective = pyo.Objective(expr=2 * model.x + 3 * model.y, sense=pyo.maximize)
    return model


def solve_example(
    model: pyo.ConcreteModel | None = None, solver: str = "gurobi"
) -> pyo.SolverResults:
    """Solve `model` (or a fresh example model) with `solver` via Pyomo's solver interface.

    `solver` is any Pyomo-registered solver name — e.g. "gurobi", "appsi_highs",
    "cbc" — which is what makes comparing backends a one-argument change
    rather than a rewrite. See `src/solver/README.md`.
    """
    model = model if model is not None else build_example_model()
    solver_interface = pyo.SolverFactory(solver)
    return solver_interface.solve(model)
