"""Archive hierarchy baselines, hybrid predictions and metrics from one YAML experiment."""

import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
import yaml

from src.forecasting.dataset import cycle_calendar, load_demand_base
from src.forecasting.harness import _now, update_record
from src.forecasting.hierarchy_config import BASELINE_WINDOW, HierarchySettings, level_keys
from src.forecasting.hierarchy_evaluation import evaluate_hierarchy
from src.forecasting.hierarchy_features import BLOCKS, feature_columns
from src.persistence.provenance import git_output as _git_output
from src.persistence.provenance import package_version as _package_version
from src.persistence.provenance import sha256_file as _sha256

DEFAULT_CONFIG = "configs/experiments/hierarchical_level/smoke.yaml"


def load_scenarios(path: str | None, calendar: pd.DataFrame) -> pd.DataFrame | None:
    """Load explicit layouts; alternative dates remain unscored."""
    if path is None:
        return None
    frame = pd.read_csv(path, dtype={"CICLOS": str, "cd_setor": str, "scenario_id": str})
    required = {"scenario_id", "CICLOS", "cd_setor", "window_start", "cycle_days"}
    if required - set(frame.columns):
        raise ValueError(f"scenario CSV requires {sorted(required)}")
    frame["window_start"] = pd.to_datetime(frame["window_start"])
    frame["cycle_days"] = pd.to_numeric(frame["cycle_days"])
    frame = frame.drop(columns="opening_date", errors="ignore").merge(
        calendar[["CICLOS", "opening_date"]], on="CICLOS", how="left", validate="many_to_one"
    )
    validate_scenarios(frame)
    return frame


def validate_scenarios(frame: pd.DataFrame) -> None:
    if frame.isna().any().any():
        raise ValueError("scenario inputs must be complete and reference known cycles")
    if frame.duplicated(["scenario_id", "CICLOS", "cd_setor"]).any():
        raise ValueError("duplicate scenario-sector-cycle")
    valid_days = frame["cycle_days"].gt(0) & frame["cycle_days"].mod(1).eq(0)
    if not (valid_days & frame["window_start"].ge(frame["opening_date"])).all():
        raise ValueError(
            "scenario windows need positive integer lengths and start after cycle opening"
        )


def prepare_record(path: Path, config: dict, settings: HierarchySettings) -> tuple[Path, dict]:
    directory = Path(config["paths"]["run_dir"])
    base = Path(config["paths"]["base_csv"])
    manifest = {
        "run_id": uuid4().hex,
        "kind": "hierarchical_level",
        "started_at": _now(),
        "config_sha256": _sha256(path),
        "data": {"path": str(base.resolve()), "sha256": _sha256(base)},
        "effective_settings": settings.model_dump(mode="json"),
        "history_policy": "only cycles closed strictly before target first opening",
        "scoring": "primary city metrics require actual > 0; other levels include all points",
        "city_identity_keys": level_keys("city", settings.city_grouping),
        "city_activity_definition": "positive observed demand in the evaluated city-identity-cycle",
        "all_points_diagnostic": "zero-demand predictions retained and scored separately",
        "code": {
            "commit": _git_output("rev-parse", "HEAD"),
            "status": _git_output("status", "--porcelain"),
        },
        "packages": {p: _package_version(p) for p in ["numpy", "pandas", "lightgbm", "scipy"]},
    }
    if settings.scenarios_csv:
        manifest["scenarios_sha256"] = _sha256(Path(settings.scenarios_csv))
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(path, directory / "config.yaml")
    _write_features(directory, settings)
    update_record(directory, manifest, "running")
    return directory, manifest


def _write_features(directory: Path, settings: HierarchySettings) -> None:
    record = {
        "city_identity_keys": level_keys("city", settings.city_grouping),
        "available_blocks": BLOCKS,
        "selected_features": feature_columns(settings.learning.features),
        "ets_inputs": ["items", "cycle_days"],
        "baseline_inputs": ["items"],
        "baseline_window": BASELINE_WINDOW,
        "model_layout": {
            "root": "ETS with shared alpha and duration exponent",
            "distribution_internal": "one pooled LightGBM regression on shares per level",
            "reconciliation_internal": (
                f"pooled LightGBM {settings.learning.volume_objective} on items per level"
            ),
            "reconciliation": "nonnegative WLS scaled by training standard deviation",
        },
    }
    (directory / "features.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def _save(result, directory: Path) -> None:
    for name in [
        "predictions",
        "comparison",
        "metrics_by_cycle",
        "folds",
        "comparison_all_points",
        "metrics_by_cycle_all_points",
    ]:
        frame = getattr(result, name).copy()
        if name == "folds":
            frame["training_cycles"] = frame["training_cycles"].map(json.dumps)
        frame.to_csv(directory / f"{name}.csv", index=False)
    (directory / "fitted_models.json").write_text(
        json.dumps(result.fitted, indent=2, allow_nan=False), encoding="utf-8"
    )


def run(path: str | Path = DEFAULT_CONFIG) -> pd.DataFrame:
    source = Path(path)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    settings = HierarchySettings.model_validate(config["settings"])
    directory, manifest = prepare_record(source, config, settings)
    try:
        raw = load_demand_base(config["paths"]["base_csv"])
        scenarios = load_scenarios(settings.scenarios_csv, cycle_calendar(raw))
        result = evaluate_hierarchy(raw, settings, scenarios)
        _save(result, directory)
        update_record(
            directory,
            manifest,
            "complete",
            n_origins=len(result.folds),
            n_predictions=len(result.predictions),
        )
    except Exception as error:
        update_record(directory, manifest, "failed", error=str(error))
        raise
    print(f"[hierarchy] outputs: {directory.resolve()}", flush=True)
    return result.comparison


if __name__ == "__main__":
    print(run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG).to_string(index=False))
