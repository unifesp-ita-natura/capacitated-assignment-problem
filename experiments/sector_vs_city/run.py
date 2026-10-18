"""Backtest each candidate per sector and per sector-city, then score both as sector totals."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.forecasting.comparison import compare
from src.forecasting.dataset import (
    CYCLE_KEYS,
    SERIES_KEY,
    build_item_panel,
    load_demand_base,
    sector_cycle_attributes,
)
from src.forecasting.evaluation import EvaluationResult, OriginResult
from src.forecasting.harness import (
    errors_by_candidate,
    evaluate_all,
    load_config,
    prepare_record,
    update_record,
    write_csv,
)

DEFAULT_CONFIG_PATH = "configs/experiments/sector_vs_city/all.yaml"
SECTOR_LABEL = "[setor]"
CITY_SUM_LABEL = "[cidades→setor]"


def relabel(result: EvaluationResult, label: str) -> EvaluationResult:
    return EvaluationResult(f"{result.candidate_name} {label}", result.origins)


def city_sum(
    city: EvaluationResult, sector: EvaluationResult, sector_items: pd.Series
) -> EvaluationResult:
    """The city forecasts summed per (sector, cycle), scored against the sector's real total.

    The actual is the sector's whole total, so a city the candidate could
    not predict (no history yet) costs the sum its items instead of being
    dropped. Each fold keeps the sector run's MASE scale, so both versions
    are divided by the same naive error.
    """
    scales = {origin.origin_cycle: origin.naive_in_sample_mae for origin in sector.origins}
    origins = []
    for origin in city.origins:
        scored = origin.scored.groupby(CYCLE_KEYS, as_index=False)["items_pred"].sum()
        keys = pd.MultiIndex.from_frame(scored[CYCLE_KEYS])
        scored["actual"] = sector_items.reindex(keys).to_numpy()
        scored = scored.assign(
            **{SERIES_KEY: scored["cd_setor"]},
            abs_error=(scored["actual"] - scored["items_pred"]).abs(),
            origin_cycle=origin.origin_cycle,
        )
        origins.append(
            OriginResult(
                origin_cycle=origin.origin_cycle,
                target_cycles=origin.target_cycles,
                scored=scored,
                missing=origin.missing,
                naive_in_sample_mae=scales[origin.origin_cycle],
            )
        )
    return EvaluationResult(f"{city.candidate_name} {CITY_SUM_LABEL}", origins)


def _execute(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_demand_base(config["paths"]["base_csv"])
    sector_panel = build_item_panel(raw, by_city=False)
    city_panel = build_item_panel(raw)
    print(
        f"[sector_vs_city] {sector_panel[SERIES_KEY].nunique()} sectors, "
        f"{city_panel[SERIES_KEY].nunique()} sector-city series",
        flush=True,
    )
    print("[sector_vs_city] per sector:", flush=True)
    sector_results = evaluate_all(config, sector_panel)
    print("[sector_vs_city] per sector-city:", flush=True)
    city_results = evaluate_all(config, city_panel)

    sector_items = sector_panel.set_index(CYCLE_KEYS)["items"]
    results = [relabel(result, SECTOR_LABEL) for result in sector_results] + [
        city_sum(city, sector, sector_items)
        for city, sector in zip(city_results, sector_results, strict=True)
    ]
    # Both versions as sector totals, sliceable in the dashboard like any run.
    sector_rows = errors_by_candidate(results).merge(
        sector_cycle_attributes(raw), on=CYCLE_KEYS, how="left"
    )
    return compare(results), sector_rows, errors_by_candidate(city_results)


def run(config_path: str | Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    config = load_config(config_path)
    directory, manifest = prepare_record(config_path, config)
    print(f"[sector_vs_city] run directory: {directory.resolve()}", flush=True)
    try:
        table, sector_rows, city_rows = _execute(config)
        write_csv(table, directory / "comparison.csv")
        write_csv(sector_rows, directory / "predictions.csv")
        write_csv(city_rows, directory / "city_predictions.csv")
        update_record(directory, manifest, "complete", n_predictions=len(sector_rows))
    except Exception as error:
        update_record(directory, manifest, "failed", error=str(error))
        raise
    return table


if __name__ == "__main__":
    comparison = run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH)
    print()
    columns = ["candidate", "mae_common", "wmape_common", "bias_pct_common", "n_scored_common"]
    print(comparison[columns].to_string(index=False))
