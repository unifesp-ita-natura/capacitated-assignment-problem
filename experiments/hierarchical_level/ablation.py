"""Replay leave-one-block-out hierarchy experiments on shared closed-cycle feature tables."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.hierarchical_level.run import _write_features, prepare_record
from src.forecasting.dataset import cycle_calendar, load_demand_base
from src.forecasting.harness import update_record
from src.forecasting.hierarchy import (
    HierarchyResult,
    _learned_predictions,
    _validate_history,
    _validate_query,
)
from src.forecasting.hierarchy_config import HierarchySettings
from src.forecasting.hierarchy_data import (
    aggregate_nodes,
    history_panel,
    known_roster,
    scenario_panel,
)
from src.forecasting.hierarchy_evaluation import (
    _actual_nodes,
    _observed_windows,
    _score,
    origin_plan,
    summarize,
)
from src.forecasting.hierarchy_features import training_and_query
from src.forecasting.hierarchy_reconciliation import assert_coherent


@dataclass
class FoldTables:
    """Keep the identical query, labels and past-only features for every ablation."""

    query: pd.DataFrame
    actual: pd.DataFrame
    training: pd.DataFrame
    target: pd.DataFrame


def variant_settings(settings: HierarchySettings) -> dict[str, HierarchySettings]:
    """Keep model settings fixed while removing exactly one selected feature block."""
    variants = {"full": settings}
    for block in settings.learning.features:
        learning = settings.learning.model_copy(
            update={"features": [name for name in settings.learning.features if name != block]}
        )
        variants[f"without_{block}"] = settings.model_copy(update={"learning": learning})
    return variants


def fold_tables(
    raw: pd.DataFrame, calendar: pd.DataFrame, plan: dict, settings: HierarchySettings
) -> FoldTables:
    as_of, cycle = plan["as_of"], plan["target_cycle"]
    available = raw[raw["data_pedido"].lt(as_of)]
    cut = calendar.assign(is_complete=calendar["is_complete"] & calendar["closing_date"].lt(as_of))
    past = history_panel(available, cut, plan["training_cycles"])
    roster = known_roster(past)
    history = aggregate_nodes(past, settings.levels, settings.city_grouping)
    windows = _observed_windows(raw, calendar, cycle)
    query = aggregate_nodes(
        scenario_panel(roster, windows), settings.levels, settings.city_grouping
    )
    actual = _actual_nodes(raw, calendar, cycle, roster, settings.levels, settings.city_grouping)
    _validate_query(query, settings.levels)
    _validate_history(history, query["opening_date"].min())
    training, target = training_and_query(history, query)
    return FoldTables(query, actual, training, target)


def learned_only(tables: FoldTables, settings: HierarchySettings) -> HierarchyResult:
    """Use the production learned-model path without recomputing unchanged moving averages."""
    result = HierarchyResult()
    _learned_predictions(tables.training, tables.target, settings, result)
    for name in ["distribution_ets_lgbm", "reconciliation_ets_lgbm"]:
        assert_coherent(tables.query, result.predictions[name], settings.levels[-1])
    return result


def _prepare_variant(directory: Path, config: dict, settings: HierarchySettings) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = {
        "settings": settings.model_dump(mode="json"),
        "paths": {"base_csv": config["paths"]["base_csv"], "run_dir": str(directory)},
    }
    (directory / "config.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8"
    )
    _write_features(directory, settings)


def _save_fold(
    directory: Path,
    tables: FoldTables,
    result: HierarchyResult,
    plan: dict,
    settings: HierarchySettings,
) -> None:
    for candidate, values in result.predictions.items():
        scored = _score(tables.query, tables.actual, values, True, settings.city_grouping).assign(
            candidate=candidate,
            scenario_id="historical",
            CICLOS=plan["target_cycle"],
            as_of=plan["as_of"],
            origin_cycle=plan["training_cycles"][-1],
        )
        path = directory / "predictions.csv"
        scored.to_csv(path, mode="a", header=not path.exists(), index=False)


def _evaluate_variants(
    raw: pd.DataFrame, settings: HierarchySettings, variants: dict, directory: Path
) -> pd.DataFrame:
    calendar = cycle_calendar(raw)
    plans = origin_plan(calendar, settings)
    if not plans:
        raise ValueError("not enough closed history for ablation")
    fitted = {name: [] for name in variants}
    for plan in plans:
        print(f"[ablation] cycle={plan['target_cycle']} variants={len(variants)}", flush=True)
        tables = fold_tables(raw, calendar, plan, settings)
        _evaluate_fold(tables, plan, variants, directory, fitted)
    for name, records in fitted.items():
        (directory / "variants" / name / "fitted_models.json").write_text(
            json.dumps(records, indent=2, allow_nan=False), encoding="utf-8"
        )
    return pd.DataFrame(plans)


def _evaluate_fold(
    tables: FoldTables, plan: dict, variants: dict, directory: Path, fitted: dict
) -> None:
    for name, settings in variants.items():
        result = learned_only(tables, settings)
        _save_fold(directory / "variants" / name, tables, result, plan, settings)
        fitted[name].extend([{**fit, "CICLOS": plan["target_cycle"]} for fit in result.fitted])


def _summaries(
    directory: Path, variants: dict, folds: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons, cycles = [], []
    for name in variants:
        variant_dir = directory / "variants" / name
        predictions = pd.read_csv(variant_dir / "predictions.csv", dtype={"CICLOS": str})
        frames = _variant_summaries(predictions)
        for filename, frame in frames.items():
            frame.to_csv(variant_dir / f"{filename}.csv", index=False)
        comparisons.append(frames["comparison"].assign(variant=name))
        cycles.append(frames["metrics_by_cycle"].assign(variant=name))
    archived = folds.copy()
    archived["training_cycles"] = archived["training_cycles"].map(json.dumps)
    archived.to_csv(directory / "folds.csv", index=False)
    return pd.concat(comparisons, ignore_index=True), pd.concat(cycles, ignore_index=True)


def _variant_summaries(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    keys = ["level", "candidate"]
    cycle_keys = [*keys, "scenario_id", "CICLOS"]
    return {
        "comparison": summarize(predictions, keys),
        "metrics_by_cycle": summarize(predictions, cycle_keys),
        "comparison_all_points": summarize(predictions, keys, city_active_only=False),
        "metrics_by_cycle_all_points": summarize(predictions, cycle_keys, city_active_only=False),
    }


def effects(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    reference = frame[frame.variant.eq("full")].drop(columns="variant")
    paired = frame[frame.variant.ne("full")].merge(reference, on=keys, suffixes=("", "_full"))
    for metric in ["mae", "wmape_pct", "bias_pct"]:
        paired[f"{metric}_delta"] = paired[metric] - paired[f"{metric}_full"]
    paired["mae_reduction_pct"] = -100 * paired.mae_delta / paired.mae_full
    return paired


def _check_reference(
    directory: Path, reference: Path, comparison: pd.DataFrame, manifest: dict
) -> None:
    original = json.loads((reference / "manifest.json").read_text())
    _check_reference_settings(original, manifest)
    current = comparison[comparison.variant.eq("full")]
    paired = current.merge(
        pd.read_csv(reference / "comparison.csv"),
        on=["level", "candidate"],
        suffixes=("", "_reference"),
        validate="one_to_one",
    )
    if len(paired) != len(current):
        raise ValueError("reference does not contain every full-model level and candidate")
    columns = ["mae", "wmape_pct", "bias_pct", "n_points", "n_missing"]
    np.testing.assert_allclose(
        paired[columns], paired[[f"{c}_reference" for c in columns]], rtol=1e-9, atol=1e-9
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(directory / "folds.csv"), pd.read_csv(reference / "folds.csv")
    )


def _check_reference_settings(original: dict, current: dict) -> None:
    if original["data"]["sha256"] != current["data"]["sha256"]:
        raise ValueError("reference data hash differs")
    if _model_settings(original) != _model_settings(current):
        raise ValueError("full-model settings differ from the archived reference")
    if original["status"] != "complete":
        raise ValueError("reference run is incomplete")


def _model_settings(record: dict) -> dict:
    """Ignore retired baseline-only options when checking learned-model references."""
    return {
        key: value
        for key, value in record["effective_settings"].items()
        if key not in {"comparison", "baseline_windows"}
    }


def _save_analysis(directory: Path, comparison: pd.DataFrame, cycles: pd.DataFrame) -> None:
    comparison.to_csv(directory / "comparison.csv", index=False)
    cycles.to_csv(directory / "metrics_by_cycle.csv", index=False)
    paired = effects(comparison, ["level", "candidate"])
    columns = ["n_points", "n_missing", "n_total_points", "n_excluded_points"]
    np.testing.assert_array_equal(paired[columns], paired[[f"{column}_full" for column in columns]])
    paired.to_csv(directory / "ablation_effects.csv", index=False)
    keys = ["level", "candidate", "scenario_id", "CICLOS"]
    effects(cycles, keys).to_csv(directory / "ablation_effects_by_cycle.csv", index=False)


def run(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    settings = HierarchySettings.model_validate(config["settings"])
    if settings.scenarios_csv is not None:
        raise ValueError("ablation currently supports only observed historical scenarios")
    directory, manifest = prepare_record(source, config, settings)
    variants = variant_settings(settings)
    manifest["experiment"] = (
        "leave_one_feature_block_out; only learned candidates; shared fold tables"
    )
    manifest["variants"] = {name: value.learning.features for name, value in variants.items()}
    try:
        for name, variant in variants.items():
            _prepare_variant(directory / "variants" / name, config, variant)
        raw = load_demand_base(config["paths"]["base_csv"])
        folds = _evaluate_variants(raw, settings, variants, directory)
        comparison, cycles = _summaries(directory, variants, folds)
        _save_analysis(directory, comparison, cycles)
        _check_reference(directory, Path(config["paths"]["reference_run"]), comparison, manifest)
        update_record(
            directory, manifest, "complete", n_origins=len(folds), n_variants=len(variants)
        )
    except Exception as error:
        update_record(directory, manifest, "failed", error=str(error))
        raise
    print(f"[ablation] outputs: {directory.resolve()}", flush=True)
    return comparison


if __name__ == "__main__":
    print(run(sys.argv[1]).to_string(index=False))
