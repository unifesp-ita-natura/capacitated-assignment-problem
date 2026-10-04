"""Gap percentual do SA contra a referência do MIP (ótimo se fechou, senão o limite inferior)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

INFEASIBLE_GAP = 100.0  # gap atribuído a uma execução que termina inviável
MIN_REFERENCE = 1e-9  # abaixo disso o gap relativo não tem significado
OPTIMAL = "optimal"


@dataclass(frozen=True)
class MipReference:
    """Denominador do gap de uma instância e de onde ele veio."""

    instance: str
    value: float | None  # ótimo (UB) se o MIP fechou, senão o limite inferior
    kind: str  # "optimal" | "lower_bound" | "missing"
    mip_gap: float | None  # gap residual do MIP, reportado sempre
    incumbent: float | None  # melhor solução viável do MIP (UB), fechado ou não


def mip_reference(record: Mapping) -> MipReference:
    """Referência de gap a partir do registro do MIP (o JSON de `mip_optima/`)."""
    closed = record.get("termination") == OPTIMAL
    value = record.get("upper_bound") if closed else record.get("lower_bound")
    kind = "missing" if value is None else ("optimal" if closed else "lower_bound")
    return MipReference(
        instance=record["instance"],
        value=value,
        kind=kind,
        mip_gap=record.get("gap"),
        incumbent=record.get("upper_bound"),
    )


def compute_gap(objective: float, feasible: bool, reference: float | None) -> float:
    """100 * (obj_sa - ref) / ref se a execução é viável; `INFEASIBLE_GAP` se não é.

    Devolve NaN quando não há referência utilizável (MIP sem limite, ou limite
    ~0): nesse caso o gap relativo não é definido e não deve entrar em médias.
    """
    if not feasible:
        return INFEASIBLE_GAP
    if reference is None or reference <= MIN_REFERENCE:
        return math.nan
    return 100.0 * (objective - reference) / reference
