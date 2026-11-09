"""Build the real-data problem instance for a given horizon length from the pipeline config."""

from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.forecasting.dataset import load_demand_base
from src.pipeline.forecast_stage import forecast_horizon, plan_horizon
from src.solver.inputs import ProblemInstance
from src.solver.real_inputs import build_real_instance


def build_instance_for_cycles(
    config_path: Path, n_cycles: int, max_sectors: int | None
) -> ProblemInstance:
    """Forecast `n_cycles` ahead with the pipeline config and return the resulting instance."""
    config = load_config(config_path)
    pipeline = config.pipeline.model_copy(update={"horizon_cycles": n_cycles})
    raw = load_demand_base(pipeline.base_csv)
    plan = plan_horizon(raw, n_cycles, pipeline.history_cycles)
    forecast = forecast_horizon(raw, plan, pipeline, config.forecast.params)
    return build_real_instance(
        forecast.daily,
        raw,
        plan.targets,
        as_is_cycle=forecast.training_cycles[-1],
        max_churn=pipeline.max_churn,
        capacity_multiplier=pipeline.capacity_multiplier,
        max_sectors=max_sectors,
    ).instance
