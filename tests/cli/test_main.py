"""Tests for the CLI entrypoint — solves for real against the bundled HiGHS backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.cli.main import main


@patch("src.cli.main.load_config")
def test_main_prints_status(mock_load_config, capsys):
    mock_config = MagicMock()
    mock_config.project_name = "capacitated-assignment-problem"
    mock_config.solver = "appsi_highs"
    mock_load_config.return_value = mock_config

    main()

    captured = capsys.readouterr()
    assert "capacitated-assignment-problem" in captured.out
    assert "ok" in captured.out
