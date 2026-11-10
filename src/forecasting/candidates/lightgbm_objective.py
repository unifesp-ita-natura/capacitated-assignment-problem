"""Pooled LightGBM with a selectable training objective (Tweedie or Quantile)."""

from __future__ import annotations

from src.config.schema import LightGBMObjectiveParams
from src.forecasting.candidates.lightgbm import _PooledLightGBMCandidate
from src.forecasting.model import REGISTRY, ForecastCandidate


class _PooledLightGBMObjectiveCandidate(_PooledLightGBMCandidate):
    """Pooled LightGBM that swaps the L1 objective for Tweedie or Quantile."""

    def __init__(self, params: LightGBMObjectiveParams) -> None:
        super().__init__(params)
        self.name = params.label or f"lightgbm_{params.objective}"

    def _settings(self) -> dict:
        settings = super()._settings()
        settings["objective"] = self._params.objective
        if self._params.objective == "tweedie":
            settings["tweedie_variance_power"] = self._params.tweedie_variance_power
        elif self._params.objective == "quantile":
            settings["alpha"] = self._params.quantile_alpha
        return settings


@REGISTRY.register("lightgbm_objective")
def build_lightgbm_objective(params: LightGBMObjectiveParams) -> ForecastCandidate:
    """Build the pooled LightGBM candidate with a Tweedie or Quantile objective."""
    return _PooledLightGBMObjectiveCandidate(params)
