"""Dados do problema de atribuição indexados pra avaliar uma atribuição sem passar pelo Pyomo."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.solver.inputs import ProblemInstance, ProjectedDemand, effective_cd_sector_shares


@dataclass(frozen=True)
class AssignmentProblem:
    """Instância no formato do MIP, com a demanda indexada por (setor, combinação)."""

    sectors: tuple[int, ...]
    cd_sectors: dict[int, tuple[int, ...]]  # CD -> setores atendidos (S_c, partição de S)
    sector_cd: dict[int, int]  # setor -> CD principal (agrupa as janelas do Relax-and-Fix)
    sector_shares: dict[int, dict[int, float]]  # setor -> {CD: fração da demanda atendida pelo CD}
    daily_capacity: Mapping[tuple[int, int], float]  # (CD, dia) -> Cap
    as_is: dict[int, int]  # setor -> combinação atual (As-Is)
    max_churn: float  # M: máximo de setores que podem mudar
    days_by_cycle: dict[int, tuple[int, ...]]  # ciclo do horizonte -> dias
    demand_by_pair: dict[
        tuple[int, int], tuple[tuple[int, float], ...]
    ]  # (s, d) -> ((dia, qtd), ...)


def build_problem(instance: ProblemInstance) -> AssignmentProblem:
    """Indexa os dados do `ProblemInstance` pro avaliador da matheurística.

    Ignora `instance.combinations`/`instance.days` (não usados fora da
    construção do modelo Pyomo); `instance.current_assignment` já é o As-Is
    denso (setor -> combinação).
    """
    sectors = tuple(instance.sectors)
    cd_sectors = {cd: tuple(members) for cd, members in instance.cd_sectors.items()}
    sector_cd = _sector_cd(sectors, cd_sectors)
    return AssignmentProblem(
        sectors=sectors,
        cd_sectors=cd_sectors,
        sector_cd=sector_cd,
        sector_shares=_sector_shares(instance),
        daily_capacity=instance.daily_capacity,
        as_is=dict(instance.current_assignment),
        max_churn=float(instance.max_churn),
        days_by_cycle={h: tuple(days) for h, days in instance.days_by_cycle.items()},
        demand_by_pair=_index_demand(instance.projected_demand),
    )


def _sector_shares(instance: ProblemInstance) -> dict[int, dict[int, float]]:
    """Setor -> {CD: share}, invertendo os shares efetivos da instância (partição rígida = 1.0)."""
    sector_shares: dict[int, dict[int, float]] = defaultdict(dict)
    for cd, shares in effective_cd_sector_shares(instance).items():
        for sector, share in shares.items():
            sector_shares[sector][cd] = share
    return dict(sector_shares)


def _sector_cd(sectors: Sequence[int], cd_sectors: Mapping[int, Sequence[int]]) -> dict[int, int]:
    """Setor -> CD, exigindo que cada setor pertença a exatamente um CD."""
    sector_cd: dict[int, int] = {}
    for cd, members in cd_sectors.items():
        for sector in members:
            if sector in sector_cd:
                raise ValueError(f"setor {sector} aparece em mais de um CD")
            sector_cd[sector] = cd
    missing = set(sectors) - set(sector_cd)
    if missing:
        raise ValueError(f"setores sem CD: {sorted(missing)}")
    return sector_cd


def _index_demand(
    projected_demand: ProjectedDemand,
) -> dict[tuple[int, int], tuple[tuple[int, float], ...]]:
    """(setor, combinação) -> pares (dia, demanda), só com demanda não nula."""
    grouped: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    for (sector, day, combination), qty in projected_demand.items():
        if qty:
            grouped[(sector, combination)].append((day, float(qty)))
    return {pair: tuple(items) for pair, items in grouped.items()}
