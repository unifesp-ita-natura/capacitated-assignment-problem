"""Pooled LightGBM candidate: one model across all sectors, implementing the Strategy directly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.schema import LightGBMParams
from src.forecasting.features import (
    SECTOR_FEATURE,
    build_features,
    build_training_table,
    feature_columns,
    lag_columns,
)
from src.forecasting.model import PREDICTION_COLUMNS, REGISTRY, ForecastCandidate


class _PooledLightGBMCandidate:
    """Strategy implemented directly rather than through `per_sector`.

    Every other candidate so far is a function over one sector's series, so
    the Adapter fits it. This one is the opposite: it trains a single model
    on all sectors pooled together, which is the whole reason to expect it
    to beat the per-sector models on this base — a sector has a handful of
    cycles, the pooled table has thousands of rows. That's why it takes the
    panel-shaped interface directly.
    """

    def __init__(self, params: LightGBMParams) -> None:
        self.name = f"lightgbm:{params.n_estimators}x{params.num_leaves}"
        self._params = params

    @property
    def _columns(self) -> list[str]:
        return feature_columns(self._params.lags, self._params.rolling_windows)

    def fit_predict(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        features, target = build_training_table(
            history, self._params.lags, self._params.rolling_windows
        )
        if features.empty:
            # Not enough cycles in the training window to build a single
            # complete lag row — the harness counts every target as missing
            # rather than this candidate inventing values.
            return pd.DataFrame(columns=PREDICTION_COLUMNS)

        sectors = pd.CategoricalDtype(sorted(history[SECTOR_FEATURE].unique()))
        model = self._fit(features, target, sectors)
        return self._forecast(model, history, targets, sectors)

    def _fit(self, features: pd.DataFrame, target: pd.Series, sectors: pd.CategoricalDtype):
        # LightGBM's native API rather than its scikit-learn wrapper: the
        # wrapper requires scikit-learn, a heavy dependency this project
        # would otherwise not need.
        import lightgbm as lgb

        dataset = lgb.Dataset(
            _as_model_frame(features, sectors),
            label=target,
            categorical_feature=[SECTOR_FEATURE],
            free_raw_data=False,
        )
        return lgb.train(
            {
                # L1 rather than LightGBM's default L2: the harness ranks
                # candidates by MAE (metrics.PRIMARY_METRIC), so the model is
                # trained on the same loss it will be judged by.
                "objective": "regression_l1",
                "learning_rate": self._params.learning_rate,
                "num_leaves": self._params.num_leaves,
                "min_data_in_leaf": self._params.min_child_samples,
                "seed": self._params.random_state,
                "verbosity": -1,
            },
            dataset,
            num_boost_round=self._params.n_estimators,
        )

    def _forecast(
        self,
        model,
        history: pd.DataFrame,
        targets: pd.DataFrame,
        sectors: pd.CategoricalDtype,
    ) -> pd.DataFrame:
        """Predict each target cycle in turn, feeding each prediction back in as the next lag.

        Multi-step forecasting is recursive: at `horizon` 2 the second
        cycle's `lag_1` is the first cycle's *prediction*, since its actual
        value is exactly what the harness is withholding.
        """
        working = history[["cd_setor", "CICLOS", "items", "opening_date"]].copy()
        predicted_frames = []

        for cycle in _chronological_cycles(targets):
            working = pd.concat([working, _pending_rows(targets, cycle)], ignore_index=True)
            rows = self._predictable_rows(working, cycle)
            if rows.empty:
                continue

            values = np.clip(model.predict(_as_model_frame(rows[self._columns], sectors)), 0, None)
            predicted_frames.append(
                pd.DataFrame(
                    {
                        "cd_setor": rows["cd_setor"].to_numpy(),
                        "CICLOS": cycle,
                        "items_pred": values,
                    }
                )
            )
            working = _fill_predictions(working, cycle, rows["cd_setor"], values)

        if not predicted_frames:
            return pd.DataFrame(columns=PREDICTION_COLUMNS)
        return pd.concat(predicted_frames, ignore_index=True)[PREDICTION_COLUMNS]

    def _predictable_rows(self, working: pd.DataFrame, cycle: str) -> pd.DataFrame:
        """Rows for `cycle` that have at least one usable lag.

        LightGBM handles missing features natively, so a partially-observed
        sector is still predictable; a sector with no prior cycle at all is
        not, and is left out so the harness reports it as missing instead.
        """
        featured = build_features(working, self._params.lags, self._params.rolling_windows)
        rows = featured[featured["CICLOS"] == cycle]
        has_any_lag = rows[lag_columns(self._params.lags)].notna().any(axis=1)
        return rows[has_any_lag]


def _as_model_frame(features: pd.DataFrame, sectors: pd.CategoricalDtype) -> pd.DataFrame:
    """Cast the sector id to a fixed categorical so train and predict share one encoding."""
    frame = features.copy()
    frame[SECTOR_FEATURE] = frame[SECTOR_FEATURE].astype(sectors)
    return frame


def _chronological_cycles(targets: pd.DataFrame) -> list[str]:
    return targets.sort_values("opening_date")["CICLOS"].drop_duplicates().tolist()


def _pending_rows(targets: pd.DataFrame, cycle: str) -> pd.DataFrame:
    """The target cycle's rows with `items` blank — the value being predicted, not yet known."""
    rows = targets[targets["CICLOS"] == cycle][["cd_setor", "CICLOS", "opening_date"]].copy()
    rows["items"] = np.nan
    return rows


def _fill_predictions(
    working: pd.DataFrame, cycle: str, sectors: pd.Series, values: np.ndarray
) -> pd.DataFrame:
    """Write this cycle's predictions into the working panel, so the next cycle's lags see them."""
    predicted = pd.Series(values, index=sectors.to_numpy())
    is_cycle = working["CICLOS"] == cycle
    working.loc[is_cycle, "items"] = working.loc[is_cycle, "cd_setor"].map(predicted)
    return working


@REGISTRY.register("lightgbm")
def build_lightgbm(params: LightGBMParams) -> ForecastCandidate:
    """Build the pooled gradient-boosting candidate the config selects."""
    return _PooledLightGBMCandidate(params)
