"""Generic provenance primitives for run manifests: file hashes, git state, package versions."""

from __future__ import annotations

import hashlib
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
