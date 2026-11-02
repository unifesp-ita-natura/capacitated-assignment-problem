"""Generic provenance primitives for run manifests: file hashes, git state, package versions."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Hex digest of `path`'s contents, read in chunks so large files don't load into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str, cwd: str | Path | None = None) -> str | None:
    """Stripped stdout of `git <args>`, or None if git or the repository is unavailable."""
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(package: str) -> str | None:
    """Installed version of `package`, or None if it isn't installed."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def code_provenance(cwd: str | Path | None = None) -> dict:
    """Current git commit and working-tree status, for recording what code produced a run."""
    status = git_output("status", "--porcelain", cwd=cwd)
    return {
        "revision": git_output("rev-parse", "HEAD", cwd=cwd),
        "working_tree_status": status.splitlines() if status else [],
    }


def save_manifest(path: str | Path, record: dict) -> None:
    """Write a run manifest as readable, NaN-free UTF-8 JSON."""
    Path(path).write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
