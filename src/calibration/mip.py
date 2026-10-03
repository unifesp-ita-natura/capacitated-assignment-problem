"""Resolve o MIP de referência com HiGHS (wrapper do MIP intocado) e extrai limites e solução."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import pyomo.environ as pyo

from src.calibration.config import InstanceSpec, MipSettings, TightenSettings
from src.calibration.instances import CalibrationInstance, ForecastChoice, build_instance
from src.solver.mip.block_assignment import build_block_assignment_model

OPTIMAL = "optimal"


@dataclass(frozen=True)
class MipOutcome:
    """Resultado do MIP numa instância: status, limites, gap residual e a atribuição x."""

    instance: str
    solver: str
    termination: str
    objective: float | None
    lower_bound: float | None
    upper_bound: float | None
    gap: float | None  # (UB - LB) / |UB|, o gap residual do MIP
    time: float
    time_limit: float
    x: dict[int, int]  # setor -> combinação da melhor solução do MIP

    @property
    def closed(self) -> bool:
        """Se o HiGHS provou otimalidade (dentro de `mip_rel_gap`)."""
        return self.termination == OPTIMAL

    @property
    def has_incumbent(self) -> bool:
        """Se o HiGHS encontrou ao menos uma solução viável."""
        return self.upper_bound is not None

    def to_dict(self) -> dict:
        """Forma serializável em JSON (chaves de `x` viram string)."""
        record = asdict(self)
        record["x"] = {str(s): d for s, d in self.x.items()}
        return record


def finite_or_none(value: float | None) -> float | None:
    """Converte ±inf/NaN (limites ausentes no Pyomo) em None."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def relative_gap(lower_bound: float | None, upper_bound: float | None) -> float | None:
    """Gap residual (UB - LB) / |UB|; None se faltar algum limite ou UB for zero."""
    if lower_bound is None or upper_bound is None or upper_bound == 0:
        return None
    return (upper_bound - lower_bound) / abs(upper_bound)


def assignment_from_model(model: pyo.ConcreteModel) -> dict[int, int]:
    """Setor -> combinação escolhida (x[s, d] = 1) numa solução carregada no modelo."""
    return {s: d for (s, d), var in model.x.items() if (var.value or 0.0) > 0.5}


def solve_mip(
    instance: CalibrationInstance, settings: MipSettings, time_limit: float
) -> MipOutcome:
    """Monta o modelo de `block_assignment` e o resolve com o solver e as opções dados."""
    model = build_block_assignment_model(**instance.mip_kwargs())
    solver = pyo.SolverFactory(settings.solver)
    start = time.perf_counter()
    results = solver.solve(
        model,
        timelimit=time_limit,
        options={"mip_rel_gap": settings.mip_rel_gap},
        load_solutions=False,
    )
    elapsed = time.perf_counter() - start
    lower = finite_or_none(results.problem.lower_bound)
    upper = finite_or_none(results.problem.upper_bound)
    x = _load_incumbent(solver, model, upper)
    return MipOutcome(
        instance=instance.name,
        solver=settings.solver,
        termination=str(results.solver.termination_condition),
        objective=upper,
        lower_bound=lower,
        upper_bound=upper,
        gap=relative_gap(lower, upper),
        time=elapsed,
        time_limit=time_limit,
        x=x,
    )


def _load_incumbent(solver, model: pyo.ConcreteModel, upper: float | None) -> dict[int, int]:
    """Carrega a melhor solução no modelo (se houver) e devolve a atribuição dela."""
    if upper is None:
        return {}
    solver.load_vars()
    return assignment_from_model(model)


def probe_feasible(
    name: str, spec: InstanceSpec, choice: ForecastChoice, search: TightenSettings
) -> bool:
    """Se o MIP acha solução viável para `spec` dentro do tempo de sondagem."""
    instance = build_instance(name, spec, choice)
    outcome = solve_mip(instance, MipSettings(), search.probe_time_limit)
    return outcome.has_incumbent


def min_feasible_multiplier(
    name: str, spec: InstanceSpec, choice: ForecastChoice, search: TightenSettings
) -> float:
    """Menor `capacity_multiplier` (a menos da resolução da bissecção) com MIP viável.

    Assume viabilidade monotônica no multiplicador: se a capacidade m é viável,
    qualquer m' > m também é. Sondagem sem solução no tempo conta como inviável,
    o que torna o resultado conservador (nunca abaixo do aperto viável).
    """
    low, high = search.low, search.high
    for _ in range(search.steps):
        middle = (low + high) / 2
        probe = spec.model_copy(update={"capacity_multiplier": middle})
        low, high = (low, middle) if probe_feasible(name, probe, choice, search) else (middle, high)
    return high
