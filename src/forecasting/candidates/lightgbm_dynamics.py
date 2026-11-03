"""Pooled LightGBM plus derived momentum and calendar features, as a separate candidate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.schema import LightGBMDynamicsParams
from src.forecasting.candidates.lightgbm import _candidate_name, _PooledLightGBMCandidate
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
            _add_momentum(featured)
        if self._params.calendar_quarter:
            featured[_QUARTER_COLUMN] = featured["opening_date"].dt.quarter
        return featured


def _add_momentum(featured: pd.DataFrame) -> None:
    """Add lag-difference and lag-to-mean ratio columns in place."""
    if not {"lag_1", "lag_2", "rolling_mean_3"} <= set(featured.columns):
        raise ValueError("momentum needs lags [1, 2] and rolling window 3 in the config")
    lag1, lag2, mean3 = featured["lag_1"], featured["lag_2"], featured["rolling_mean_3"]
    featured["mom_diff_1_2"] = lag1 - lag2
    featured["mom_ratio_1_mean3"] = (lag1 / mean3.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )


@REGISTRY.register("lightgbm_dynamics")
def build_lightgbm_dynamics(params: LightGBMDynamicsParams) -> ForecastCandidate:
    """Build the pooled LightGBM candidate with momentum and calendar features."""
    return _PooledLightGBMDynamicsCandidate(params)
