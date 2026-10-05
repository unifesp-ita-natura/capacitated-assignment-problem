"""Import every shipped candidate module so it registers itself into `model.REGISTRY`."""

from __future__ import annotations

from src.forecasting.candidates import (  # noqa: F401
    arima,
    cycle_factor,
    ets,
    lightgbm,
    naive,
)
