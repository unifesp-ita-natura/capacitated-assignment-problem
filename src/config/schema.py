"""Pydantic schema for the top-level run configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class Paths(BaseModel):
    instances: str = "data/instances"
    processed: str = "data/processed"
    results: str = "results"


class TrainWindow(BaseModel):
    lookback_periods: int = 52
    horizon_periods: int = 4


class NaiveParams(BaseModel):
    model: Literal["naive"] = "naive"
    strategy: Literal["last_value", "seasonal_naive", "mean"] = "last_value"
    season_length: int | None = None

    @model_validator(mode="after")
    def _season_length_valid(self):
        if self.strategy == "seasonal_naive":
            if not isinstance(self.season_length, int) or self.season_length <= 0:
                raise ValueError(
                    "season_length must be a positive integer when strategy is 'seasonal_naive'"
                )
        else:
            self.season_length = None
        return self


class ArimaParams(BaseModel):
    model: Literal["arima"] = "arima"
    order: tuple[int, int, int] = (1, 0, 0)  # (p, d, q)
    seasonal_order: tuple[int, int, int, int] | None = None  # (P, D, Q, s)
    trend: Literal["n", "c", "t", "ct"] | None = None


class LightGBMParams(BaseModel):
    model: Literal["lightgbm"] = "lightgbm"
    n_estimators: int = 100
    learning_rate: float = 0.1
    num_leaves: int = 31
    lags: list[int] = [1, 2, 3, 4]
    rolling_windows: list[int] = [3, 6]
    # LightGBM's own default (20) leaves almost nothing to split on with this
    # base's training-set size (~4k pooled rows, see the experiment README).
    min_child_samples: int = 5
    random_state: int = 0
    # Lags of the order and volume counts the panel carries alongside items.
    # Empty keeps the model on the item series alone, which is how every
    # result before the panel_order_counts experiment was produced — see
    # docs/plans/forecasting-feature-roadmap.md, stage 1.
    companion_lags: list[int] = []
    # Overrides the generated candidate name. The comparison table keys rows
    # by name, so variants that differ only in a field the generated name
    # doesn't carry (rolling windows, lags) need one to be tellable apart.
    label: str | None = None
    # "level" fits the item count itself; "ratio" fits items divided by the
    # sector's running mean, so the model predicts a correction on top of
    # the naive:mean benchmark instead of relearning 633 sector sizes.
    target: Literal["level", "ratio"] = "level"
    # Stochastic and penalty terms. At their defaults LightGBM uses every
    # row and every column on every tree, which makes random_state inert —
    # so seeds only start to inform once these move off 1.0.
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0
    bagging_freq: int = 0
    lambda_l2: float = 0.0
    # Stops boosting when the most recent cycles of the TRAINING window
    # stop improving. Those cycles are held out of the fit, so the number
    # of rounds is chosen without ever touching the cycle being forecast.
    early_stopping_rounds: int | None = None
    validation_cycles: int = 1


class CycleFactorParams(BaseModel):
    model: Literal["cycle_factor"] = "cycle_factor"
    # "last" repeats the most recent observed factor; "mean" averages the
    # last `window` of them; "all" averages every factor seen so far, which
    # amounts to a constant correction for the benchmark's bias.
    strategy: Literal["last", "mean", "all"] = "last"
    window: int = 1


ForecastParams = Annotated[
    NaiveParams | ArimaParams | LightGBMParams | CycleFactorParams,
    Field(discriminator="model"),
]


class ForecastConfig(BaseModel):
    params: ForecastParams = NaiveParams()
    train_window: TrainWindow = TrainWindow()


class RunConfig(BaseModel):
    project_name: str = "experiments-template"
    modeling: str = "pyomo"
    solver: str = "gurobi"
    paths: Paths = Paths()

    forecast: ForecastConfig = ForecastConfig()
