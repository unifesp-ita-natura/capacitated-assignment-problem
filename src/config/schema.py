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


class ETSParams(BaseModel):
    model: Literal["ets"] = "ets"
    # ETS(error, trend, seasonal) in Hyndman's taxonomy. The defaults give
    # ETS(A,N,N) — simple exponential smoothing, a weighted mean that sits
    # between naive:last_value (alpha = 1) and naive:mean (alpha -> 0).
    error: Literal["add", "mul"] = "add"
    trend: Literal["add", "mul"] | None = None
    damped_trend: bool = False
    seasonal: Literal["add", "mul"] | None = None
    seasonal_periods: int | None = None
    # One smoothing weight shared by every sector instead of one fitted per
    # sector: a number fixes it, "pooled" picks the one that minimises the
    # one-step-ahead error summed over every sector's training history.
    # Only for ETS(A,N,N), where the point forecast depends on alpha alone.
    alpha: float | Literal["pooled"] | None = None
    # How a cycle's items scale with its length, items ~ cycle_days ** exponent:
    # 1 is "items per day times days". "pooled" picks it with alpha from the
    # training history. Fixing it needs a shared alpha; the per-sector ETS
    # always estimates it (model.window_scaled).
    cycle_days_exponent: float | Literal["pooled"] = "pooled"

    @model_validator(mode="after")
    def _components_valid(self):
        for check in (
            _trend_problem,
            _alpha_model_problem,
            _alpha_range_problem,
            _exponent_problem,
            _season_problem,
        ):
            if problem := check(self):
                raise ValueError(problem)
        return self


def _trend_problem(params: ETSParams) -> str | None:
    if params.damped_trend and params.trend is None:
        return "damped_trend needs a trend"
    return None


def _alpha_model_problem(params: ETSParams) -> str | None:
    if params.alpha is not None and (params.trend or params.seasonal or params.error != "add"):
        return "a shared alpha is only supported for ETS(A,N,N)"
    return None


def _alpha_range_problem(params: ETSParams) -> str | None:
    if isinstance(params.alpha, float) and not 0 < params.alpha <= 1:
        return "alpha must be in (0, 1]"
    return None


def _exponent_problem(params: ETSParams) -> str | None:
    if params.cycle_days_exponent != "pooled" and params.alpha is None:
        return "a fixed cycle_days_exponent needs a shared alpha"
    return None


def _season_problem(params: ETSParams) -> str | None:
    if params.seasonal is not None and (params.seasonal_periods or 0) < 2:
        return "seasonal_periods must be >= 2 when seasonal is set"
    return None


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
    NaiveParams | ArimaParams | ETSParams | LightGBMParams | CycleFactorParams,
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
