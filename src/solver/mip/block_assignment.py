"""Pyomo formulation for the sector-to-Bloco/Subloco assignment problem (desafio Natura)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pyomo.environ as pyo

ProjectedDemand = Mapping[tuple[int, int, int], float]  # (sector, day, combination) -> qty


def build_block_assignment_model(
    sectors: Iterable[int],
    combinations: Iterable[int],
    days: Iterable[int],
    cd_sectors: Mapping[int, Iterable[int]],
    daily_capacity: Mapping[tuple[int, int], float],
    projected_demand: ProjectedDemand,
    current_assignment: Mapping[tuple[int, int], int],
    max_churn: float,
) -> pyo.ConcreteModel:
    """Assign each sector to one Bloco/Subloco to level daily demand across the horizon.

    `daily_capacity` and `current_assignment` are keyed `(cd, day)` and
    `(sector, combination)` respectively; both default to 0 for missing keys.
    """
    cd_sectors = {cd: list(members) for cd, members in cd_sectors.items()}

    model = pyo.ConcreteModel()
    _add_sets(model, sectors, combinations, days, cd_sectors)
    _add_variables(model)
    _add_objective(model)
    _add_assignment_constraint(model)
    _add_demand_range_constraints(model, projected_demand)
    _add_capacity_constraints(model, cd_sectors, daily_capacity, projected_demand)
    _add_churn_constraint(model, current_assignment, max_churn)
    return model


def solve_block_assignment(model: pyo.ConcreteModel, solver: str = "gurobi") -> pyo.SolverResults:
    """Solve `model` with `solver` via Pyomo's solver interface.

    `solver` is any Pyomo-registered solver name, matching the pattern in
    `src/solver/mip/example.py`.
    """
    solver_interface = pyo.SolverFactory(solver)
    return solver_interface.solve(model)


def _add_sets(
    model: pyo.ConcreteModel,
    sectors: Iterable[int],
    combinations: Iterable[int],
    days: Iterable[int],
    cd_sectors: Mapping[int, list[int]],
) -> None:
    model.S = pyo.Set(initialize=list(sectors))
    model.D = pyo.Set(initialize=list(combinations))
    model.A = pyo.Set(initialize=list(days))
    model.C = pyo.Set(initialize=list(cd_sectors))


def _add_variables(model: pyo.ConcreteModel) -> None:
    model.x = pyo.Var(model.S, model.D, within=pyo.Binary)
    model.z_max = pyo.Var(within=pyo.Reals)
    model.z_min = pyo.Var(within=pyo.Reals)


def _add_objective(model: pyo.ConcreteModel) -> None:
    model.objective = pyo.Objective(expr=model.z_max - model.z_min, sense=pyo.minimize)


def _add_assignment_constraint(model: pyo.ConcreteModel) -> None:
    def rule(m: pyo.ConcreteModel, s: int) -> bool:
        return sum(m.x[s, d] for d in m.D) == 1

    model.single_assignment = pyo.Constraint(model.S, rule=rule)


def _add_demand_range_constraints(
    model: pyo.ConcreteModel, projected_demand: ProjectedDemand
) -> None:
    def total_demand(m: pyo.ConcreteModel, a: int) -> pyo.Expression:
        return sum(projected_demand.get((s, a, d), 0.0) * m.x[s, d] for s in m.S for d in m.D)

    model.demand_upper_bound = pyo.Constraint(
        model.A, rule=lambda m, a: m.z_max >= total_demand(m, a)
    )
    model.demand_lower_bound = pyo.Constraint(
        model.A, rule=lambda m, a: m.z_min <= total_demand(m, a)
    )


def _add_capacity_constraints(
    model: pyo.ConcreteModel,
    cd_sectors: Mapping[int, list[int]],
    daily_capacity: Mapping[tuple[int, int], float],
    projected_demand: ProjectedDemand,
) -> None:
    def rule(m: pyo.ConcreteModel, c: int, a: int) -> bool:
        cd_demand = sum(
            projected_demand.get((s, a, d), 0.0) * m.x[s, d] for s in cd_sectors[c] for d in m.D
        )
        return cd_demand <= daily_capacity.get((c, a), 0.0)

    model.cd_capacity = pyo.Constraint(model.C, model.A, rule=rule)


def _add_churn_constraint(
    model: pyo.ConcreteModel,
    current_assignment: Mapping[tuple[int, int], int],
    max_churn: float,
) -> None:
    def rule(m: pyo.ConcreteModel) -> bool:
        changed = sum(
            current_assignment.get((s, d), 0)
            + (1 - 2 * current_assignment.get((s, d), 0)) * m.x[s, d]
            for s in m.S
            for d in m.D
        )
        return changed <= 2 * max_churn

    model.churn_limit = pyo.Constraint(rule=rule)
