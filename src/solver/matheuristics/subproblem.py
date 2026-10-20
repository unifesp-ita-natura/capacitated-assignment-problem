"""Prepara o MIP pra cada subproblema (fixa/relaxa x) e o resolve com um solver Pyomo."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Collection, Mapping

import pyomo.environ as pyo

logger = logging.getLogger(__name__)

# Resolve `model` no lugar (carrega a solução em `model.x`) com um limite de tempo opcional
# em segundos; devolve False se o solver não achou nenhuma solução viável.
SubproblemSolver = Callable[[pyo.ConcreteModel, float | None], bool]


def apply_variable_modes(
    model: pyo.ConcreteModel, fixed: Mapping[int, int], binary: Collection[int]
) -> None:
    """Define como cada `x[s, d]` entra no próximo subproblema.

    Setores em `fixed` (setor -> combinação) têm `x` congelado em 0/1; os de
    `binary` ficam livres e binários; todos os outros ficam relaxados em [0, 1].
    """
    for (sector, combination), var in model.x.items():
        _set_mode(var, sector, combination, fixed, binary)


def _set_mode(
    var: pyo.Var,
    sector: int,
    combination: int,
    fixed: Mapping[int, int],
    binary: Collection[int],
) -> None:
    if sector in fixed:
        var.fix(1 if fixed[sector] == combination else 0)
        return
    var.unfix()
    var.domain = pyo.Binary if sector in binary else pyo.UnitInterval


def read_assignment(model: pyo.ConcreteModel, sectors: Collection[int]) -> dict[int, int]:
    """Combinação escolhida (a de maior `x`) de cada setor em `sectors`, na solução carregada."""
    return {sector: _chosen_combination(model, sector) for sector in sectors}


def _chosen_combination(model: pyo.ConcreteModel, sector: int) -> int:
    return max(model.D, key=lambda d: pyo.value(model.x[sector, d], exception=False) or 0.0)


class PyomoSubproblemSolver:
    """Solver de subproblemas via Pyomo (HiGHS por padrão), reaproveitado entre as chamadas.

    Usa `timelimit` e `options` do `appsi_highs`, como em `src/calibration/mip.py`. O solver é
    persistente: o modelo vai pro HiGHS na 1a chamada e, nas seguintes, só as mudanças nos
    limites/domínios de `x` são repassadas. Isso exige tratar variáveis fixadas como variáveis
    (não como parâmetros), senão o Pyomo reprocessaria todas as restrições a cada subproblema,
    que é o custo dominante do Relax-and-Fix.
    """

    def __init__(self, solver_name: str = "appsi_highs", mip_rel_gap: float = 1e-4) -> None:
        self._solver_name = solver_name
        self._mip_rel_gap = mip_rel_gap
        self._solver = None

    def __call__(self, model: pyo.ConcreteModel, time_limit: float | None) -> bool:
        solver = self._persistent_solver()
        results = solver.solve(
            model,
            **_solve_kwargs(self._solver_name, time_limit, self._mip_rel_gap),
            load_solutions=False,
        )
        if not _has_incumbent(results):
            logger.warning("subproblema sem solução: %s", results.solver.termination_condition)
            return False
        solver.load_vars()
        return True

    def _persistent_solver(self):
        if self._solver is None:
            self._solver = _new_persistent_solver(self._solver_name)
        return self._solver


def pyomo_subproblem_solver(
    solver_name: str = "appsi_highs", mip_rel_gap: float = 1e-4
) -> SubproblemSolver:
    """Solver de subproblemas via Pyomo (HiGHS por padrão); veja `PyomoSubproblemSolver`."""
    return PyomoSubproblemSolver(solver_name, mip_rel_gap)


def _new_persistent_solver(solver_name: str):
    solver = pyo.SolverFactory(solver_name)
    update_config = getattr(solver, "update_config", None)
    if update_config is not None:
        update_config.treat_fixed_vars_as_params = False
    return solver


GAP_OPTION = {
    "appsi_highs": "mip_rel_gap",
    "appsi_gurobi": "MIPGap",
}


def mip_gap_option(solver_name: str) -> str:
    """Nome da opção de gap relativo do MIP no solver (cada um usa o seu)."""
    return GAP_OPTION.get(solver_name, "mip_rel_gap")


def _solve_kwargs(solver_name: str, time_limit: float | None, mip_rel_gap: float) -> dict:
    kwargs: dict = {"options": {mip_gap_option(solver_name): mip_rel_gap}}
    if time_limit is not None:
        kwargs["timelimit"] = time_limit
    return kwargs


def _has_incumbent(results: pyo.SolverResults) -> bool:
    """Se o solver terminou com alguma solução viável (limite superior finito)."""
    upper = results.problem.upper_bound
    return upper is not None and math.isfinite(upper)
