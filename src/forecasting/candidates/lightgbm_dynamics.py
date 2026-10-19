"""Pooled LightGBM plus derived momentum and calendar features, as a separate candidate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.schema import LightGBMDynamicsParams
from src.forecasting.candidates.lightgbm import (
    _OUTPUT_COLUMNS,
    _candidate_name,
    _category_types,
    _PooledLightGBMCandidate,
    _trainable_rows,
)
from src.forecasting.features import (
    SECTOR_FEATURE,
    build_features,
    complete_feature_frame,
    lag_columns,
)
from src.forecasting.model import REGISTRY, ForecastCandidate

_MOMENTUM_COLUMNS = ["mom_diff_1_2", "mom_ratio_1_mean3"]
_QUARTER_COLUMN = "opening_quarter"


class _PooledLightGBMDynamicsCandidate(_PooledLightGBMCandidate):
    """Pooled LightGBM that also feeds derived momentum and calendar columns."""

    def __init__(self, params: LightGBMDynamicsParams) -> None:
        super().__init__(params)
        if params.label is None:
            extras = []
            if params.momentum:
                extras.append("mom")
            if params.calendar_quarter:
                extras.append("quarter")
            self.name = f"{_candidate_name(params)}+{'+'.join(extras) or 'none'}"

    @property
    def _derived(self) -> list[str]:
        columns = []
        if self._params.momentum:
            columns += _MOMENTUM_COLUMNS
        if self._params.calendar_quarter:
            columns.append(_QUARTER_COLUMN)
        return columns

    @property
    def _columns(self) -> list[str]:
        return [*super()._columns, *self._derived]

    def _derive(self, featured: pd.DataFrame) -> pd.DataFrame:
        """Add the momentum and quarter columns to an already-featured frame."""
        featured = featured.copy()
        if self._params.momentum:
            lag1, lag2 = featured.get("lag_1"), featured.get("lag_2")
            mean3 = featured.get("rolling_mean_3")
            if lag1 is None or lag2 is None or mean3 is None:
                raise ValueError(
                    "momentum needs lags [1, 2] and rolling window 3 in the config"
                )
            featured["mom_diff_1_2"] = lag1 - lag2
            featured["mom_ratio_1_mean3"] = (lag1 / mean3.replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan
            )
        if self._params.calendar_quarter:
            featured[_QUARTER_COLUMN] = featured["opening_date"].dt.quarter
        return featured

    def fit_predict(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        complete = _trainable_rows(
            self._derive(
                complete_feature_frame(
                    history,
                    self._params.lags,
                    self._params.rolling_windows,
                    self._params.companion_lags,
                    self._params.categorical_features,
                )
            ),
            self._params.target,
        )
        if complete.empty:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)

        sectors = _category_types(history, [SECTOR_FEATURE, *self._params.categorical_features])
        model = self._fit(complete, sectors)
        return self._forecast(model, history, targets, sectors)

    def _predictable_rows(self, working: pd.DataFrame, cycle: str) -> pd.DataFrame:
        featured = self._derive(
            build_features(
                working,
                self._params.lags,
                self._params.rolling_windows,
                self._params.companion_lags,
            )
        )
        rows = featured[featured["CICLOS"] == cycle]
        has_any_lag = rows[lag_columns(self._params.lags)].notna().any(axis=1)
        return _trainable_rows(rows[has_any_lag], self._params.target)


@REGISTRY.register("lightgbm_dynamics")
def build_lightgbm_dynamics(params: LightGBMDynamicsParams) -> ForecastCandidate:
    """Build the pooled LightGBM candidate with momentum and calendar features."""
    return _PooledLightGBMDynamicsCandidate(params)
