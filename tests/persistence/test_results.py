"""Tests for the SolveResult schema and CSV persistence."""

from __future__ import annotations

import csv
import subprocess
from unittest.mock import patch

from src.persistence import SolveResult, current_git_commit, write_results_csv


def test_write_results_csv_round_trips(tmp_path):
    results = [
        SolveResult(
            model_name="example",
            solver="gurobi",
            status="ok",
            termination_condition="optimal",
            objective=20.0,
            wall_time_seconds=0.01,
            git_commit="abc123",
        ),
        SolveResult(
            model_name="example",
            solver="appsi_highs",
            status="ok",
            termination_condition="optimal",
            objective=20.0,
            wall_time_seconds=0.02,
            git_commit="abc123",
        ),
    ]
    out_path = tmp_path / "results.csv"

    write_results_csv(results, out_path)

    with open(out_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["solver"] == "gurobi"
    assert rows[1]["solver"] == "appsi_highs"
    assert rows[0]["git_commit"] == "abc123"


def test_current_git_commit_returns_a_sha_inside_a_git_repo():
    commit = current_git_commit()

    assert commit is not None
    assert len(commit) == 40


def test_current_git_commit_returns_none_outside_a_git_repo():
    with patch("src.persistence.results.subprocess.run", side_effect=FileNotFoundError):
        assert current_git_commit() is None


def test_current_git_commit_returns_none_when_git_fails():
    error = subprocess.CalledProcessError(returncode=128, cmd=["git", "rev-parse", "HEAD"])
    with patch("src.persistence.results.subprocess.run", side_effect=error):
        assert current_git_commit() is None


def test_solve_result_defaults_git_commit_from_current_repo():
    result = SolveResult(
        model_name="example",
        solver="gurobi",
        status="ok",
        termination_condition="optimal",
        objective=20.0,
        wall_time_seconds=0.01,
    )

    assert result.git_commit == current_git_commit()
