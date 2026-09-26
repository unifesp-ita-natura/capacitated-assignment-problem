"""Pooled LightGBM candidate: one model across all sectors, implementing the Strategy directly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.schema import LightGBMParams
from src.forecasting.features import (
    COMPANION_SERIES,
    LEVEL_REFERENCE,
    SECTOR_FEATURE,
    build_features,
    complete_feature_frame,
    feature_columns,
    lag_columns,
)
from src.forecasting.model import PREDICTION_COLUMNS, REGISTRY, ForecastCandidate

# What the recursive forecast has to carry forward cycle by cycle, before
# any companion series the configuration adds.
_BASE_PANEL_COLUMNS = ("cd_setor", "CICLOS", "items", "opening_date")


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
        self.name = _candidate_name(params)
        self._params = params

    @property
    def _columns(self) -> list[str]:
        return feature_columns(
            self._params.lags, self._params.rolling_windows, self._params.companion_lags
        )

    @property
    def _panel_columns(self) -> list[str]:
        """Panel columns the recursive forecast carries forward, not just the target series."""
        if not self._params.companion_lags:
            return list(_BASE_PANEL_COLUMNS)
        return [*_BASE_PANEL_COLUMNS, *COMPANION_SERIES]

    def fit_predict(self, history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        complete = _trainable_rows(
            complete_feature_frame(
                history,
                self._params.lags,
                self._params.rolling_windows,
                self._params.companion_lags,
            ),
            self._params.target,
        )
        if complete.empty:
            # Not enough cycles in the training window to build a single
            # complete lag row — the harness counts every target as missing
            # rather than this candidate inventing values.
            return pd.DataFrame(columns=PREDICTION_COLUMNS)

        sectors = pd.CategoricalDtype(sorted(history[SECTOR_FEATURE].unique()))
        model = self._fit(complete, sectors)
        return self._forecast(model, history, targets, sectors)

    def _target_of(self, complete: pd.DataFrame) -> pd.Series:
        """What the model is asked to fit: the item count, or its ratio to the sector's level."""
        if self._params.target == "level":
            return complete["items"].copy()
        return (complete["items"] / complete[LEVEL_REFERENCE]).copy()

    def _to_items(self, predicted: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
        """Undo the target transform, so what leaves the candidate is always in items."""
        if self._params.target == "level":
            return predicted
        return predicted * rows[LEVEL_REFERENCE].to_numpy()

    def _settings(self) -> dict:
        return {
            # L1 rather than LightGBM's default L2: the harness ranks
            # candidates by MAE (metrics.PRIMARY_METRIC), so the model is
            # trained on the same loss it will be judged by.
            "objective": "regression_l1",
            "learning_rate": self._params.learning_rate,
            "num_leaves": self._params.num_leaves,
            "min_data_in_leaf": self._params.min_child_samples,
            "feature_fraction": self._params.feature_fraction,
            "bagging_fraction": self._params.bagging_fraction,
            "bagging_freq": self._params.bagging_freq,
            "lambda_l2": self._params.lambda_l2,
            "seed": self._params.random_state,
            "verbosity": -1,
        }

    def _fit(self, complete: pd.DataFrame, sectors: pd.CategoricalDtype):
        """Train on the complete rows, holding out the training window's last cycles if asked."""
        # LightGBM's native API rather than its scikit-learn wrapper: the
        # wrapper requires scikit-learn, a heavy dependency this project
        # would otherwise not need.
        import lightgbm as lgb

        fit_rows, validation_rows = self._split_for_early_stopping(complete)
        dataset = self._dataset(lgb, fit_rows, sectors)
        if validation_rows is None:
            return lgb.train(self._settings(), dataset, num_boost_round=self._params.n_estimators)
        return lgb.train(
            self._settings(),
            dataset,
            num_boost_round=self._params.n_estimators,
            valid_sets=[self._dataset(lgb, validation_rows, sectors)],
            callbacks=[lgb.early_stopping(self._params.early_stopping_rounds, verbose=False)],
        )

    def _dataset(self, lgb, rows: pd.DataFrame, sectors: pd.CategoricalDtype):
        return lgb.Dataset(
            _as_model_frame(rows[self._columns], sectors),
            label=self._target_of(rows),
            categorical_feature=[SECTOR_FEATURE],
            free_raw_data=False,
        )

    def _split_for_early_stopping(
        self, complete: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Last cycles of the training window become the validation set, the rest the fit set.

        The split is by cycle, not by row, so a sector never appears on both
        sides of it. It never reaches past the training window, so the
        number of rounds is still chosen without seeing the target cycle.
        Too few cycles to spare one means boosting simply runs the full
        `n_estimators`.
        """
        if self._params.early_stopping_rounds is None:
            return complete, None
        cycles = sorted(complete["opening_date"].unique())
        if len(cycles) <= self._params.validation_cycles:
            return complete, None
        cutoff = cycles[-self._params.validation_cycles]
        held_out = complete["opening_date"] >= cutoff
        return complete[~held_out], complete[held_out]

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
        working = history[self._panel_columns].copy()
        predicted_frames = []

        for cycle in _chronological_cycles(targets):
            pending = _pending_rows(targets, cycle, self._panel_columns)
            working = pd.concat([working, pending], ignore_index=True)
            rows = self._predictable_rows(working, cycle)
            if rows.empty:
                continue

            predicted = model.predict(_as_model_frame(rows[self._columns], sectors))
            values = np.clip(self._to_items(predicted, rows), 0, None)
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
        featured = build_features(
            working,
            self._params.lags,
            self._params.rolling_windows,
            self._params.companion_lags,
        )
        rows = featured[featured["CICLOS"] == cycle]
        has_any_lag = rows[lag_columns(self._params.lags)].notna().any(axis=1)
        return _trainable_rows(rows[has_any_lag], self._params.target)


def _trainable_rows(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Drop rows a ratio target can't use — no level reference means nothing to divide by."""
    if target == "level":
        return frame
    return frame[frame[LEVEL_REFERENCE] > 0]


def _as_model_frame(features: pd.DataFrame, sectors: pd.CategoricalDtype) -> pd.DataFrame:
    """Cast the sector id to a fixed categorical so train and predict share one encoding."""
    frame = features.copy()
    frame[SECTOR_FEATURE] = frame[SECTOR_FEATURE].astype(sectors)
    return frame


def _chronological_cycles(targets: pd.DataFrame) -> list[str]:
    return targets.sort_values("opening_date")["CICLOS"].drop_duplicates().tolist()


def _pending_rows(targets: pd.DataFrame, cycle: str, panel_columns: list[str]) -> pd.DataFrame:
    """The target cycle's rows with every measured series blank — none of them is known yet.

    Order and volume counts for the cycle being predicted are as unknown as
    the item count itself, so they stay `NaN` and only their lags, which
    reach back into observed cycles, ever reach the model.
    """
    rows = targets[targets["CICLOS"] == cycle][["cd_setor", "CICLOS", "opening_date"]].copy()
    blank = [column for column in panel_columns if column not in rows.columns]
    return rows.assign(**dict.fromkeys(blank, np.nan))


def _fill_predictions(
    working: pd.DataFrame, cycle: str, sectors: pd.Series, values: np.ndarray
) -> pd.DataFrame:
    """Write this cycle's predictions into the working panel, so the next cycle's lags see them."""
    predicted = pd.Series(values, index=sectors.to_numpy())
    is_cycle = working["CICLOS"] == cycle
    working.loc[is_cycle, "items"] = working.loc[is_cycle, "cd_setor"].map(predicted)
    return working


def _candidate_name(params: LightGBMParams) -> str:
    """Name carrying what distinguishes one configured pooled model from another in a run."""
    if params.label is not None:
        return params.label
    shape = f"lightgbm:{params.n_estimators}x{params.num_leaves}"
    if params.companion_lags:
        shape += "+orders"
    return f"{shape}:ratio" if params.target == "ratio" else shape


@REGISTRY.register("lightgbm")
def build_lightgbm(params: LightGBMParams) -> ForecastCandidate:
    """Build the pooled gradient-boosting candidate the config selects."""
    return _PooledLightGBMCandidate(params)
