"""Re-score an assignment on the realized (held-out) demand instead of the forecast."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.pipeline.explain import Scenario, describe
from src.solver import inputs as solver_inputs
from src.solver.inputs import ProblemInstance
from src.solver.matheuristics.problem import build_problem
from src.solver.real_inputs import RealInstance, cycle_spacing, sector_curves


def realized_daily(raw: pd.DataFrame, cycles: list[str]) -> pd.DataFrame:
    """Actual items per (sector, cycle, relative day), shaped like the forecast daily frame."""
    selected = raw[raw["CICLOS"].isin(cycles)]
    start = selected.groupby(["cd_setor", "CICLOS"])["Dt Abertura"].transform("min")
    frame = selected.assign(
        relative_day=(selected["data_pedido"] - start).dt.days + 1,
        items_pred=selected["total_itens_mascarado"],
    )
    return frame.groupby(["cd_setor", "CICLOS", "relative_day"], as_index=False)["items_pred"].sum()


def realized_instance(
    forecast_instance: ProblemInstance, real: RealInstance, raw: pd.DataFrame, cycles: list[str]
) -> ProblemInstance:
    """`forecast_instance` with its demand swapped for the actuals of the same cycles.

    Sectors, calendar, capacities and As-Is are untouched, so any assignment found on
    the forecast can be evaluated here; demand outside the instance's days is dropped.
    """
    curves = sector_curves(realized_daily(raw, cycles), cycles)
    curves = curves[curves["sector"].isin(real.sector_ids)]
    combo_slots = solver_inputs.build_id_maps(list(real.sector_ids))[1]
    demand = solver_inputs.build_projected_demand(
        curves, real.sector_ids, combo_slots, cycle_spacing(raw, cycles), 0
    )
    days = set(forecast_instance.days)
    return replace(
        forecast_instance, projected_demand={k: v for k, v in demand.items() if k[1] in days}
    )


def rescore(
    forecast_instance: ProblemInstance, realized: ProblemInstance, assignment: dict[int, int] | None
) -> tuple[Scenario, Scenario | None]:
    """As-Is and To-Be scenarios of the forecast-optimized assignment on the actual demand."""
    problem = build_problem(realized)
    as_is = describe(problem, realized, forecast_instance.current_assignment)
    to_be = None if assignment is None else describe(problem, realized, assignment)
    return as_is, to_be
