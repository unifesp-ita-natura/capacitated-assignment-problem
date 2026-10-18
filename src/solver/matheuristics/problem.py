"""Dados do problema de atribuição indexados pra avaliar uma atribuição sem passar pelo Pyomo."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

ProjectedDemand = Mapping[tuple[int, int, int], float]  # (setor, dia, combinação) -> qtd


@dataclass(frozen=True)
class AssignmentProblem:
    """Instância no formato do MIP, com a demanda indexada por (setor, combinação)."""

    sectors: tuple[int, ...]
    cd_sectors: dict[int, tuple[int, ...]]  # CD -> setores atendidos (S_c, partição de S)
    sector_cd: dict[int, int]  # setor -> CD que o atende
    daily_capacity: Mapping[tuple[int, int], float]  # (CD, dia) -> Cap
    as_is: dict[int, int]  # setor -> combinação atual (As-Is)
    max_churn: float  # M: máximo de setores que podem mudar
    days_by_cycle: dict[int, tuple[int, ...]]  # ciclo do horizonte -> dias
    demand_by_pair: dict[
        tuple[int, int], tuple[tuple[int, float], ...]
    ]  # (s, d) -> ((dia, qtd), ...)


def build_problem(
    sectors: Iterable[int],
    cd_sectors: Mapping[int, Iterable[int]],
    daily_capacity: Mapping[tuple[int, int], float],
    projected_demand: ProjectedDemand,
    current_assignment: Mapping[tuple[int, int], int],
    max_churn: float,
    days_by_cycle: Mapping[int, Iterable[int]],
) -> AssignmentProblem:
    """Indexa os dados do MIP (mesmos argumentos de `build_block_assignment_model`).

    `current_assignment` é o As-Is esparso do MIP, `{(setor, combinação): 1}`.
    """
    sectors = tuple(sectors)
    cd_sectors = {cd: tuple(members) for cd, members in cd_sectors.items()}
    sector_cd = _sector_cd(sectors, cd_sectors)
    return AssignmentProblem(
        sectors=sectors,
        cd_sectors=cd_sectors,
        sector_cd=sector_cd,
        daily_capacity=daily_capacity,
        as_is={s: d for (s, d), flag in current_assignment.items() if flag},
        max_churn=float(max_churn),
        days_by_cycle={h: tuple(days) for h, days in days_by_cycle.items()},
        demand_by_pair=_index_demand(projected_demand),
    )


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
