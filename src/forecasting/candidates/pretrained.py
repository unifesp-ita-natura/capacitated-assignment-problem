"""Time-series foundation models used zero-shot: Chronos-2, TimesFM-2.5 and TabPFN-TS."""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pandas as pd

from src.config.schema import PretrainedParams
from src.forecasting.model import REGISTRY, ForecastCandidate, per_series_batch


@cache
def _chronos2():
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")


def chronos2(series: list[pd.Series], horizon: int, point: str, max_context: int) -> np.ndarray:
    """Chronos-2: its median, or a mean taken over its nine deciles.

    The "mean" Chronos reports beside its quantiles is the median, so the
    mean is approximated by averaging the 10%..90% quantiles instead.
    """
    import torch

    levels = [0.5] if point == "median" else [k / 10 for k in range(1, 10)]
    contexts = [torch.tensor(s.to_numpy(dtype=float)[-max_context:]) for s in series]
    quantiles, _ = _chronos2().predict_quantiles(
        contexts, prediction_length=horizon, quantile_levels=levels
    )
    return np.stack([q.mean(dim=-1).reshape(-1).numpy() for q in quantiles])


@cache
def _timesfm(max_context: int):
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=128,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def timesfm25(series: list[pd.Series], horizon: int, point: str, max_context: int) -> np.ndarray:
    """TimesFM-2.5: quantile output 0 is the mean, output 5 the median."""
    context = 32 * math.ceil(max_context / 32)  # the model reads 32-step patches
    inputs = [s.to_numpy(dtype=float)[-max_context:] for s in series]
    _, quantiles = _timesfm(context).forecast(horizon=horizon, inputs=inputs)
    return np.asarray(quantiles[:, :horizon, 0 if point == "mean" else 5], dtype=float)


@cache
def _tabpfn(point: str):
    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
    from tabpfn_time_series.features import CalendarFeature, RunningIndexFeature

    # No AutoSeasonalFeature: cycles are ~3 weeks apart but not evenly.
    return TabPFNTSPipeline(
        tabpfn_mode=TabPFNMode.LOCAL,
        temporal_features=[RunningIndexFeature(), CalendarFeature()],
        tabpfn_output_selection=point,
    )


def tabpfn_ts(
    series: list[pd.Series], future: list[pd.Timestamp], point: str, max_context: int
) -> np.ndarray:
    """TabPFN-TS: a tabular foundation model regressing each series on its own dates."""
    context = pd.concat(
        [
            pd.DataFrame(
                {
                    "item_id": i,
                    "timestamp": s.index[-max_context:],
                    "target": s.to_numpy()[-max_context:],
                }
            )
            for i, s in enumerate(series)
        ],
        ignore_index=True,
    )
    wanted = pd.MultiIndex.from_product(
        [range(len(series)), future], names=["item_id", "timestamp"]
    )
    predicted = _tabpfn(point).predict_df(
        context_df=context, future_df=wanted.to_frame(index=False), quantiles=[0.5]
    )
    return predicted["target"].reindex(wanted).to_numpy().reshape(len(series), len(future))


@REGISTRY.register("pretrained")
def build_pretrained(params: PretrainedParams) -> ForecastCandidate:
    """Build the zero-shot candidate; weights download on first use and stay loaded."""

    def forecast(series: list[pd.Series], future: list[pd.Timestamp]) -> np.ndarray:
        if params.name == "tabpfn_ts":
            predicted = tabpfn_ts(series, future, params.point, params.max_context)
        elif params.name == "timesfm2.5":
            predicted = timesfm25(series, len(future), params.point, params.max_context)
        else:
            predicted = chronos2(series, len(future), params.point, params.max_context)
        return np.clip(predicted, 0, None)

    return per_series_batch(f"{params.name}:{params.point}", forecast)
