"""Pooled LightGBM plus static per-UF external features, as a separate candidate."""

from __future__ import annotations

from src.config.schema import LightGBMExternalParams
from src.forecasting.candidates.lightgbm import _PooledLightGBMCandidate
from src.forecasting.model import REGISTRY, ForecastCandidate


class _PooledLightGBMExternalCandidate(_PooledLightGBMCandidate):
    """Pooled LightGBM that also feeds the panel's external UF columns as numeric features."""

    def __init__(self, params: LightGBMExternalParams) -> None:
        super().__init__(params)
        self.name = params.label or f"lightgbm_external:{'+'.join(params.external_features)}"

    @property
    def _external(self) -> list[str]:
        return list(self._params.external_features)

    @property
    def _columns(self) -> list[str]:
        return [*super()._columns, *self._external]

    @property
    def _panel_columns(self) -> list[str]:
        return [*super()._panel_columns, *self._external]

    @property
    def _carried_columns(self) -> list[str]:
        return [*super()._carried_columns, *self._external]


@REGISTRY.register("lightgbm_external")
def build_lightgbm_external(params: LightGBMExternalParams) -> ForecastCandidate:
    """Build the pooled LightGBM candidate that also reads external features."""
    return _PooledLightGBMExternalCandidate(params)
