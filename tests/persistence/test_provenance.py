"""Tests for the generic run-manifest provenance helpers."""

from __future__ import annotations

import hashlib
import subprocess

from src.persistence import provenance


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes(b"items\n100\n")
    assert provenance.sha256_file(path) == hashlib.sha256(b"items\n100\n").hexdigest()


def test_git_output_returns_stripped_stdout(tmp_path):
    assert provenance.git_output("--version").startswith("git version")


def test_metadata_collection_tolerates_missing_package_and_git(monkeypatch):
    def missing_package(package):
        raise provenance.PackageNotFoundError(package)

    def missing_git(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(provenance, "version", missing_package)
    monkeypatch.setattr(provenance.subprocess, "check_output", missing_git)
    assert provenance.package_version("none") is None
    assert provenance.git_output("status") is None


def test_git_output_tolerates_a_failing_command(monkeypatch):
    def failing(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(provenance.subprocess, "check_output", failing)
    assert provenance.git_output("status") is None
