"""Tests for the CLI entrypoint — Gurobi is mocked, never called directly."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.cli.main import main


@patch("src.cli.main.solve_example")
def test_main_prints_status(mock_solve_example, capsys):
    mock_results = MagicMock()
    mock_results.solver.status = "ok"
    mock_solve_example.return_value = mock_results

    main()

    captured = capsys.readouterr()
    assert "capacitated-assignment-problem" in captured.out
    assert "ok" in captured.out
