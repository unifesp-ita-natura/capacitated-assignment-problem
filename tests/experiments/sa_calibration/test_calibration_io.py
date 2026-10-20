"""Testes da persistência da calibração: CSV de execuções, retomada, traços e tabelas."""

from __future__ import annotations

import pandas as pd

from src.calibration import io
from src.calibration.runner import RunSpec


def test_append_run_writes_header_once_and_done_keys_reads_back(tmp_path):
    path = io.runs_path(tmp_path, "screen")
    for seed in (1, 2):
        io.append_run(
            path,
            {"phase": "screen", "config_id": "a", "instance": "x", "seed": seed, "gap": 1.0},
        )

    assert path.read_text().count("phase,config_id") == 1
    assert io.done_keys(path) == {("screen", "a", "x", 1), ("screen", "a", "x", 2)}


def test_done_keys_of_a_missing_file_is_empty(tmp_path):
    assert io.done_keys(tmp_path / "nope.csv") == set()


def test_trace_roundtrip_uses_the_run_file_name(tmp_path):
    spec = RunSpec("screen", "lhs_00", {}, "base40", 3)
    path = io.trace_path(tmp_path, spec)

    io.write_trace(path, [(0, 5.0, 5.0, 100.0), (10, 4.0, 4.0, 90.0)])

    assert path.name == "lhs_00_base40_3.csv"
    assert list(io.read_trace(path).columns) == list(io.TRACE_COLUMNS)


def test_json_roundtrip(tmp_path):
    path = io.mip_path(tmp_path, "base40")

    io.write_json(path, {"instance": "base40", "objective": 1.5})

    assert io.read_json(path) == {"instance": "base40", "objective": 1.5}


def test_to_markdown_formats_numbers():
    table = io.to_markdown(pd.DataFrame({"config": ["a"], "gap": [1.23456], "it": [200000.0]}))

    assert table.splitlines() == [
        "| config | gap | it |",
        "|---|---|---|",
        "| a | 1.235 | 200,000 |",
    ]
