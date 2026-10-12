"""Run and archive a shape-only backtest configured by a single YAML file."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.forecasting.dataset import load_demand_base
from src.forecasting.shape_daily import build_shape_days
from src.forecasting.shape_evaluation import (
    ShapeCandidate,
    ShapeSplit,
    compare_shapes,
    evaluate_shapes,
)
from src.forecasting.shape_learning import BASE_FEATURES


class ShapePaths(BaseModel):
    """Input CSV and a new directory for this execution's artifacts."""

    model_config = ConfigDict(extra="forbid")
    base_csv: Path
    run_dir: Path


class ShapeConfig(BaseModel):
    """Complete shape experiment configuration independent of level settings."""

    model_config = ConfigDict(extra="forbid")
    split: ShapeSplit = Field(default_factory=ShapeSplit)
    candidates: list[ShapeCandidate] = Field(min_length=1)
    paths: ShapePaths


def _git_revision() -> str:
    """Record the current Git revision when available."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip()


def _manifest(config: ShapeConfig) -> dict:
    """Describe effective parameters, inputs and implemented predictors."""
    with config.paths.base_csv.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "git_status": subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.splitlines(),
        "python_version": sys.version,
        "package_versions": {
            name: version(name) for name in ["pandas", "numpy", "pydantic", "pyyaml"]
        },
        "lightgbm_version": _learning_version(config),
        "base_sha256": digest,
        "parameters": config.model_dump(mode="json"),
        "target_granularity": ["cd_setor", "CICLOS", "date"],
        "ranking_metric": "share_mae_pp",
        "level_for_evaluation": "actual cycle total; no level model fitted",
        "features_by_candidate": {
            candidate.name: _candidate_metadata(candidate) for candidate in config.candidates
        },
    }


def _learning_version(config: ShapeConfig) -> str | None:
    """Record the optional dependency only when a learned shape candidate is selected."""
    if any(candidate.model == "lightgbm" for candidate in config.candidates):
        return version("lightgbm")
    return None


def _candidate_metadata(candidate: ShapeCandidate) -> dict:
    """Record only predictors used by each strategy."""
    inputs = {
        "uniform": [],
        "sector": ["cd_setor", "offset", "n_days", "actual_share"],
        "lightgbm": [*BASE_FEATURES, *candidate.learning.features],
    }
    return {
        "model": candidate.model,
        "n_bins": candidate.n_bins,
        "effective_parameters": candidate.model_dump(mode="json"),
        "inputs": ["window_start", "n_days"],
        "features": inputs[candidate.model],
    }


def _organization_inputs(config: ShapeConfig, days) -> None:
    """Reject requested organization predictors absent from the input base."""
    requested = {
        feature
        for candidate in config.candidates
        if candidate.model == "lightgbm"
        for feature in candidate.learning.features
    }
    missing = requested.intersection({"CD_RE", "CD_GV"}) - set(days.columns)
    if missing:
        raise ValueError(
            f"shape input base is missing requested organization features: {sorted(missing)}"
        )


def _save_manifest(directory: Path, record: dict) -> None:
    """Write execution metadata as readable UTF-8 JSON."""
    (directory / "run_manifest.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _execute(config: ShapeConfig, directory: Path) -> None:
    """Load real demand, evaluate curves and export daily predictions and metrics."""
    days = build_shape_days(load_demand_base(config.paths.base_csv))
    _organization_inputs(config, days)
    predictions = evaluate_shapes(days, config.candidates, config.split)
    comparison = compare_shapes(predictions)
    predictions.to_csv(directory / "predictions.csv", index=False)
    comparison.to_csv(directory / "comparison.csv", index=False)
    print(comparison.to_string(index=False))


def run(config_path: str | Path) -> Path:
    """Archive one execution without overwriting previous runs."""
    contents = Path(config_path).read_text(encoding="utf-8-sig")
    config = ShapeConfig.model_validate(yaml.safe_load(contents))
    record = _manifest(config)
    directory = config.paths.run_dir
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config_snapshot.yaml").write_text(contents, encoding="utf-8")
    _save_manifest(directory, record)
    print(f"Shape output: {directory.resolve()}")
    try:
        _execute(config, directory)
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    else:
        record.update(status="completed")
    finally:
        record["finished_at"] = datetime.now(UTC).isoformat()
        _save_manifest(directory, record)
    return directory


def main() -> None:
    """Accept the path to the single shape experiment YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
