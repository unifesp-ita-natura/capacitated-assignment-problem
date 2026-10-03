"""Monta instâncias do CAP para a calibração com o gerador, o forecasting e `solver.inputs`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.calibration.config import InstanceSpec
from src.forecasting.data import forecast_items, items_per_order_by_sector
from src.forecasting.model import forecast_future_cycles
from src.forecasting.scoring import score_and_select_strategies
from src.generate import generator
from src.solver import inputs as solver_inputs


@dataclass(frozen=True)
class ForecastChoice:
    """Estratégias de previsão pedidas no YAML; None deixa a seleção automática decidir."""

    level: str | None = None
    shape: str | None = None


@dataclass(frozen=True)
class SyntheticHistory:
    """Saída do gerador: histórico sintético de uma instância, ainda em rótulos de setor."""

    sectors: list[str]
    demand_level: pd.DataFrame
    demand_shape: pd.DataFrame
    cycle_starts: list[pd.Timestamp]
    assignment: dict[str, tuple[int, int]]
    sector_to_cd: dict[str, int]


@dataclass(frozen=True)
class CalibrationInstance:
    """Uma instância pronta para o SA e para o MIP, com as estratégias de previsão usadas."""

    name: str
    sectors: list[int]
    combinations: list[int]
    days: list[int]
    cd_sectors: dict[int, list[int]]
    daily_capacity: dict[tuple[int, int], float]
    projected_demand: dict[tuple[int, int, int], float]
    current_assignment_sa: dict[int, int]
    current_assignment_mip: dict[tuple[int, int], int]
    max_churn: float
    days_by_cycle: dict[int, list[int]]
    level_strategy: str
    shape_strategy: str

    def _common_kwargs(self) -> dict:
        return {
            "sectors": self.sectors,
            "combinations": self.combinations,
            "days": self.days,
            "cd_sectors": self.cd_sectors,
            "daily_capacity": self.daily_capacity,
            "projected_demand": self.projected_demand,
            "max_churn": self.max_churn,
            "days_by_cycle": self.days_by_cycle,
        }

    def sa_kwargs(self) -> dict:
        """Argumentos nomeados de `simulated_annealing.solve`, exceto params/rng/callback."""
        return {**self._common_kwargs(), "current_assignment": self.current_assignment_sa}

    def mip_kwargs(self) -> dict:
        """Argumentos nomeados de `block_assignment.build_block_assignment_model`."""
        return {**self._common_kwargs(), "current_assignment": self.current_assignment_mip}

    def metadata(self) -> dict:
        """Resumo auditável da instância (tamanho e estratégias de previsão usadas)."""
        return {
            "name": self.name,
            "n_sectors": len(self.sectors),
            "n_combinations": len(self.combinations),
            "n_days": len(self.days),
            "n_cycles_horizon": len(self.days_by_cycle),
            "max_churn_sectors": self.max_churn,
            "level_strategy": self.level_strategy,
            "shape_strategy": self.shape_strategy,
        }


def synthesize_history(spec: InstanceSpec) -> SyntheticHistory:
    """Roda o gerador com a seed da instância, na mesma ordem de consumo do rng de `src/main.py`."""
    rng = np.random.default_rng(spec.seed)
    sectors = generator.build_sectors(spec.n_sectors)
    level, shape, starts, assignment = generator.generate_synthetic_demand(
        rng, sectors, spec.cycle_length, spec.n_cycles_history
    )
    sector_to_cd = generator.build_sector_cd_assignment(rng, sectors)
    return SyntheticHistory(sectors, level, shape, starts, assignment, sector_to_cd)


def resolve_strategies(
    history: SyntheticHistory, spec: InstanceSpec, choice: ForecastChoice
) -> tuple[str, str]:
    """Estratégias (nível, forma) efetivas: as do YAML, completadas pela seleção automática."""
    if choice.level is not None and choice.shape is not None:
        return choice.level, choice.shape
    auto_level, auto_shape = score_and_select_strategies(
        history.demand_level,
        history.demand_shape,
        history.cycle_starts,
        history.assignment,
        spec.n_cycles_history,
        spec.n_cycles_horizon,
    )
    return choice.level or auto_level, choice.shape or auto_shape


def forecast_calendar(
    history: SyntheticHistory, spec: InstanceSpec, strategies: tuple[str, str]
) -> pd.DataFrame:
    """Previsão nível+forma dos ciclos futuros em datas de calendário, já em itens."""
    forecast = forecast_future_cycles(
        history.demand_level,
        history.demand_shape,
        history.cycle_starts,
        history.assignment,
        spec.n_cycles_history,
        spec.cycle_length,
        strategies[0],
        strategies[1],
        spec.n_cycles_horizon,
    )
    return forecast_items(forecast, items_per_order_by_sector(history.demand_level))


def build_instance(
    name: str, spec: InstanceSpec, choice: ForecastChoice | None = None
) -> CalibrationInstance:
    """Gera, prevê e converte uma instância para os dicionários que o SA e o MIP esperam."""
    history = synthesize_history(spec)
    strategies = resolve_strategies(history, spec, choice or ForecastChoice())
    calendar_forecast = forecast_calendar(history, spec, strategies)
    return _to_solver_inputs(name, spec, history, calendar_forecast, strategies)


def _to_solver_inputs(
    name: str,
    spec: InstanceSpec,
    history: SyntheticHistory,
    calendar_forecast: pd.DataFrame,
    strategies: tuple[str, str],
) -> CalibrationInstance:
    """Traduz rótulos de setor/slot em ids inteiros com as funções de `src/solver/inputs.py`."""
    sector_ids, combo_slots = solver_inputs.build_id_maps(history.sectors)
    combo_ids = {slot: combo_id for combo_id, slot in combo_slots.items()}
    cycle_span = generator.cycle_span_business_days(spec.cycle_length)
    projected_demand = solver_inputs.build_projected_demand(
        calendar_forecast, sector_ids, combo_slots, cycle_span, spec.n_cycles_history + 1
    )
    days = sorted({day for _, day, _ in projected_demand})
    cd_sectors = solver_inputs.build_cd_sectors(history.sector_to_cd, sector_ids)
    return CalibrationInstance(
        name=name,
        sectors=list(sector_ids.values()),
        combinations=list(combo_slots),
        days=days,
        cd_sectors=cd_sectors,
        daily_capacity=solver_inputs.build_daily_capacity(
            list(cd_sectors), days, spec.capacity_multiplier
        ),
        projected_demand=projected_demand,
        current_assignment_sa=solver_inputs.build_current_assignment_sa(
            history.assignment, sector_ids, combo_ids
        ),
        current_assignment_mip=solver_inputs.build_current_assignment_mip(
            history.assignment, sector_ids, combo_ids
        ),
        max_churn=spec.max_churn * spec.n_sectors,
        days_by_cycle=solver_inputs.group_days_by_cycle(days, cycle_span),
        level_strategy=strategies[0],
        shape_strategy=strategies[1],
    )
