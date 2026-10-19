"""Pooled LightGBM plus static per-UF external features, as a separate candidate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.schema import LightGBMExternalParams
from src.forecasting.candidates.lightgbm import (
    _OUTPUT_COLUMNS,
    _carry_organization,
    _chronological_cycles,
    _fill_predictions,
    _pending_rows,
    _PooledLightGBMCandidate,
)
from src.forecasting.dataset import SERIES_KEY
from src.forecasting.model import QUERY_KEYS, REGISTRY, ForecastCandidate


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

    def _forecast(
        self,
        model,
        history: pd.DataFrame,
        targets: pd.DataFrame,
        sectors: dict[str, pd.CategoricalDtype],
    ) -> pd.DataFrame:
        working = history[self._panel_columns].copy()
        carried = [*self._params.categorical_features, *self._external]
        predicted_frames = []

        for cycle in _chronological_cycles(targets):
            pending = _pending_rows(targets, cycle, self._panel_columns)
            pending = _carry_organization(pending, working, carried)
            working = pd.concat([working, pending], ignore_index=True)
            rows = self._predictable_rows(working, cycle)
            if rows.empty:
                continue

            rows = self._scenario_rows(rows, targets, cycle)
            predicted = model.predict(self._model_frame(rows, sectors))
            frame = rows[QUERY_KEYS].assign(
                items_pred=np.clip(self._to_items(predicted, rows), 0, None)
            )
            predicted_frames.append(frame)
            working = _fill_predictions(
                working, cycle, frame.groupby(SERIES_KEY)["items_pred"].mean()
            )

        if not predicted_frames:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)
        return pd.concat(predicted_frames, ignore_index=True)

    def _model_frame(self, rows: pd.DataFrame, sectors: dict[str, pd.CategoricalDtype]):
        from src.forecasting.candidates.lightgbm import _as_model_frame

        return _as_model_frame(rows[self._columns], sectors)


@REGISTRY.register("lightgbm_external")
def build_lightgbm_external(params: LightGBMExternalParams) -> ForecastCandidate:
    """Build the pooled LightGBM candidate that also reads external features."""
    return _PooledLightGBMExternalCandidate(params)
