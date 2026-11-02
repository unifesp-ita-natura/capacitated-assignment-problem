"""Tests the end-to-end pipeline entry point on the small demand fixture."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config.schema import PipelineConfig, RunConfig
from src.main import run_pipeline


@pytest.mark.parametrize("family", ["mip", "matheuristic", "sa"])
def test_run_pipeline_end_to_end(family: str, tmp_path) -> None:
    config = RunConfig(
        solver="appsi_highs",
        paths={"results": str(tmp_path)},
        pipeline=PipelineConfig(
            base_csv="tests/fixtures/demand_sample.csv",
            history_cycles=8,
            horizon_cycles=2,
            solver_family=family,
            report_accuracy=False,
        ),
    )
    explanation = run_pipeline(config)
    assert explanation.to_be is not None
    assert explanation.to_be.objective <= explanation.as_is.objective + 1e-6
    expected = {"pipeline_solve.csv", "pipeline_explanation.csv", "pipeline_assignment.csv"}
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assignment = pd.read_csv(tmp_path / "pipeline_assignment.csv")
    assert assignment["to_be_slot"].notna().all()
