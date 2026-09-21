"""Monta as instâncias de calibração a partir do gerador sintético (espelha `src/main.py`)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import repo

generator = repo.generator
solver_inputs = repo.solver_inputs
forecast_items = repo.forecasting_data.forecast_items
items_per_order_by_sector = repo.forecasting_data.items_per_order_by_sector
forecast_future_cycles = repo.forecasting_model.forecast_future_cycles
score_and_select_strategies = repo.forecasting_scoring.score_and_select_strategies


@dataclass
class Instance:
    """Uma instância do CAP já no formato que `solve` espera."""

    name: str
    sectors: list[int]
    combinations: list[int]
    days: list[int]
    cd_sectors: dict[int, list[int]]
    daily_capacity: dict[tuple[int, int], float]
    projected_demand: dict[tuple[int, int, int], float]
    current_assignment_sa: dict[int, int]
    max_churn: float
    days_by_cycle: dict[int, list[int]]

    def sa_kwargs(self) -> dict:
        """Os argumentos nomeados de `simulated_annealing.solve`, menos params/rng."""
        return {
            "sectors": self.sectors,
            "combinations": self.combinations,
            "days": self.days,
            "cd_sectors": self.cd_sectors,
            "daily_capacity": self.daily_capacity,
            "projected_demand": self.projected_demand,
            "current_assignment": self.current_assignment_sa,
            "max_churn": self.max_churn,
            "days_by_cycle": self.days_by_cycle,
        }


def _forecast_calendar(
    demand_level,
    demand_shape,
    cycle_starts,
    assignment,
    n_cycles_history,
    cycle_length,
    n_cycles_horizon,
):
    """Previsão combinada nível+forma para os ciclos futuros, já em datas de calendário."""
    best_level_name, best_shape_name = score_and_select_strategies(
        demand_level, demand_shape, cycle_starts, assignment, n_cycles_history, n_cycles_horizon
    )
    return forecast_future_cycles(
        demand_level,
        demand_shape,
        cycle_starts,
        assignment,
        n_cycles_history,
        cycle_length,
        best_level_name,
        best_shape_name,
        n_cycles_horizon,
    )


def build_instance(
    name: str = "inst",
    n_sectors: int = 40,
    cycle_length: int = 21,
    n_cycles_history: int = 6,
    n_cycles_horizon: int = 2,
    max_churn: float = 0.2,
    capacity_multiplier: float = 1.0,
    seed: int = 42,
) -> Instance:
    """Gera demanda sintética, prevê os ciclos futuros e monta os dicionários do solver.

    `capacity_multiplier` aperta ou afrouxa o limite real de cada CD: com o valor
    1.0 e poucas centenas de setores a capacidade nunca chega a restringir, e
    `sector_bias`/`destination_bias`/`penalty_coefficient` ficam praticamente
    inertes — use valores pequenos (~0.003 para 40 setores) para calibrá-los.
    """
    rng = np.random.default_rng(seed)
    sectors = generator.build_sectors(n_sectors)

    demand_level, demand_shape, cycle_starts, assignment = generator.generate_synthetic_demand(
        rng, sectors, cycle_length, n_cycles_history
    )
    calendar_forecast = _forecast_calendar(
        demand_level,
        demand_shape,
        cycle_starts,
        assignment,
        n_cycles_history,
        cycle_length,
        n_cycles_horizon,
    )

    sector_ids, combo_slots = solver_inputs.build_id_maps(sectors)
    combo_ids = {slot: combo_id for combo_id, slot in combo_slots.items()}

    items_per_order = items_per_order_by_sector(demand_level)
    calendar_forecast = forecast_items(calendar_forecast, items_per_order)

    cycle_span = generator.cycle_span_business_days(cycle_length)
    projected_demand = solver_inputs.build_projected_demand(
        calendar_forecast, sector_ids, combo_slots, cycle_span, n_cycles_history + 1
    )
    days = sorted({day for _, day, _ in projected_demand})
    sector_to_cd = generator.build_sector_cd_assignment(rng, sectors)
    cd_sectors = solver_inputs.build_cd_sectors(sector_to_cd, sector_ids)

    return Instance(
        name=name,
        sectors=list(sector_ids.values()),
        combinations=list(combo_slots.keys()),
        days=days,
        cd_sectors=cd_sectors,
        daily_capacity=solver_inputs.build_daily_capacity(
            list(cd_sectors), days, capacity_multiplier
        ),
        projected_demand=projected_demand,
        current_assignment_sa=solver_inputs.build_current_assignment_sa(
            assignment, sector_ids, combo_ids
        ),
        max_churn=max_churn * n_sectors,
        days_by_cycle=solver_inputs.group_days_by_cycle(days, cycle_span),
    )
