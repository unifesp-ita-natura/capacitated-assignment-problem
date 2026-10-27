"""Import every shipped candidate module so it registers itself into `model.REGISTRY`."""

from __future__ import annotations

from src.forecasting.candidates import (  # noqa: F401
    arima,
    croston,
    cycle_factor,
    ets,
    lightgbm,
    lightgbm_dynamics,
    lightgbm_external,
    lightgbm_objective,
    naive,
    prophet,
)
