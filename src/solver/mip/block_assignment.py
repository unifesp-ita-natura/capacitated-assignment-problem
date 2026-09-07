"""Formulação em Pyomo do problema de atribuição setor -> Bloco/Subloco (desafio Natura)."""

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
    """Atribui cada setor a um Bloco/Subloco pra nivelar a demanda diária no horizonte.

    `daily_capacity` e `current_assignment` são indexados por `(cd, dia)` e
    `(setor, combinação)`, respectivamente; ambos assumem 0 pra chaves ausentes.
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
    """Resolve `model` com `solver` via a interface de solvers do Pyomo.

    `solver` é qualquer nome de solver registrado no Pyomo, seguindo o mesmo
    padrão de `src/solver/mip/example.py`.
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
    """Declara S, D, A, C (§3.1: setores, combinações, dias, CDs)."""
    model.S = pyo.Set(initialize=list(sectors))
    model.D = pyo.Set(initialize=list(combinations))
    model.A = pyo.Set(initialize=list(days))
    model.C = pyo.Set(initialize=list(cd_sectors))


def _add_variables(model: pyo.ConcreteModel) -> None:
    """Declara x (§3.3: 1 se o setor s usa a combinação d) e os limites z_max/z_min."""
    model.x = pyo.Var(model.S, model.D, within=pyo.Binary)
    model.z_max = pyo.Var(within=pyo.Reals)
    model.z_min = pyo.Var(within=pyo.Reals)


def _add_objective(model: pyo.ConcreteModel) -> None:
    """Minimiza a amplitude de demanda z_max - z_min (§3.4).

    z_max/z_min ainda não ficam presos ao máximo/mínimo diário real por
    nenhuma restrição aqui — veja `_add_demand_range_constraints` pra essa
    outra metade do truque.
    """
    model.objective = pyo.Objective(expr=model.z_max - model.z_min, sense=pyo.minimize)


def _add_assignment_constraint(model: pyo.ConcreteModel) -> None:
    """Cada setor escolhe exatamente uma combinação Bloco/Subloco (§3.5.1)."""

    def rule(m: pyo.ConcreteModel, s: int) -> bool:
        return sum(m.x[s, d] for d in m.D) == 1

    model.single_assignment = pyo.Constraint(model.S, rule=rule)


def _add_demand_range_constraints(
    model: pyo.ConcreteModel, projected_demand: ProjectedDemand
) -> None:
    """Prende z_max/z_min como teto/piso da demanda total de cada dia (§3.5.2, §3.5.3).

    Combinado com o objetivo de minimizar z_max - z_min, isso força z_max a
    virar o máximo diário real e z_min o mínimo diário real no ótimo — veja
    a explicação de epígrafe/hipógrafe na descrição do PR.
    """

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
    """Limita a demanda diária total de cada CD à sua capacidade (§3.5.4)."""

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
    """Limita a max_churn o número de setores que mudam em relação ao As-Is (§3.5.5).

    Cada par (s, d) contribui com 1 - x[s,d] se era a atribuição As-Is (0 se
    o setor manteve, 1 se abandonou) ou com x[s,d] caso contrário (1 se o
    setor escolheu essa combinação agora). Um setor que mudou contribui com
    exatamente 2 somando todos os seus pares (s, d); um que não mudou
    contribui com 0 — por isso o orçamento é 2 * max_churn, não max_churn.
    """

    def rule(m: pyo.ConcreteModel) -> bool:
        changed = sum(
            current_assignment.get((s, d), 0)
            + (1 - 2 * current_assignment.get((s, d), 0)) * m.x[s, d]
            for s in m.S
            for d in m.D
        )
        return changed <= 2 * max_churn

    model.churn_limit = pyo.Constraint(rule=rule)
