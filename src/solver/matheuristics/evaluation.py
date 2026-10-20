"""Avalia uma atribuição setor -> combinação: objetivo, capacidade excedida e orçamento de churn."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from src.solver.matheuristics.problem import AssignmentProblem

TOLERANCE = 1e-6


@dataclass(frozen=True)
class Evaluation:
    """Objetivo e viabilidade de uma atribuição, calculados direto dos dados (sem o solver)."""

    objective: float  # média, entre os ciclos, de (máx - mín) da demanda diária
    capacity_violation: float  # soma dos excessos sobre Cap_{c,a} (0 se viável)
    churn_budget_used: float  # lado esquerdo de §3.5.5 (2 por setor que mudou)
    churn_budget: float  # 2 * M

    @property
    def feasible(self) -> bool:
        """Se respeita capacidade (§3.5.4) e churn (§3.5.5) dentro da tolerância numérica."""
        within_capacity = self.capacity_violation <= TOLERANCE
        return within_capacity and self.churn_budget_used <= self.churn_budget + TOLERANCE


def evaluate_assignment(problem: AssignmentProblem, assignment: Mapping[int, int]) -> Evaluation:
    """Avalia `assignment` (setor -> combinação) com a mesma definição do MIP."""
    daily_total, cd_load = _loads(problem, assignment)
    return Evaluation(
        objective=_balance_objective(problem, daily_total),
        capacity_violation=_capacity_violation(problem, cd_load),
        churn_budget_used=_churn_budget_used(problem, assignment),
        churn_budget=2.0 * problem.max_churn,
    )


def capacity_utilization(problem: AssignmentProblem, assignment: Mapping[int, int]) -> float:
    """Maior razão carga/capacidade sobre os pares (CD, dia) com capacidade positiva.

    Acima de 1 a atribuição estoura algum CD; serve pra dimensionar o aperto de uma instância.
    """
    _, cd_load = _loads(problem, assignment)
    ratios = [
        load / problem.daily_capacity[key]
        for key, load in cd_load.items()
        if problem.daily_capacity.get(key, 0.0) > 0
    ]
    return max(ratios, default=0.0)


def _loads(
    problem: AssignmentProblem, assignment: Mapping[int, int]
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """Demanda total por dia e carga por (CD, dia) da atribuição."""
    daily_total: dict[int, float] = defaultdict(float)
    cd_load: dict[tuple[int, int], float] = defaultdict(float)
    for sector, combination in assignment.items():
        cd = problem.sector_cd[sector]
        for day, qty in problem.demand_by_pair.get((sector, combination), ()):
            daily_total[day] += qty
            cd_load[(cd, day)] += qty
    return daily_total, cd_load


def _balance_objective(problem: AssignmentProblem, daily_total: Mapping[int, float]) -> float:
    """Média, entre os ciclos do horizonte, da amplitude da demanda diária de cada ciclo."""
    ranges = [_cycle_range(days, daily_total) for days in problem.days_by_cycle.values()]
    return sum(ranges) / len(ranges)


def _cycle_range(days: tuple[int, ...], daily_total: Mapping[int, float]) -> float:
    totals = [daily_total.get(day, 0.0) for day in days]
    return max(totals) - min(totals)


def _capacity_violation(
    problem: AssignmentProblem, cd_load: Mapping[tuple[int, int], float]
) -> float:
    """P_cap: soma dos excessos de carga sobre a capacidade de cada CD em cada dia."""
    return sum(
        max(0.0, load - problem.daily_capacity.get(key, 0.0)) for key, load in cd_load.items()
    )


def _churn_budget_used(problem: AssignmentProblem, assignment: Mapping[int, int]) -> float:
    """Lado esquerdo de §3.5.5: 2 por setor que saiu do As-Is, 1 se o setor não tem As-Is."""
    return sum(_sector_churn(problem.as_is.get(s), d) for s, d in assignment.items())


def _sector_churn(as_is: int | None, chosen: int) -> float:
    if as_is is None:
        return 1.0
    return 0.0 if as_is == chosen else 2.0
