#!/usr/bin/env bash
# One-time per-clone setup: symlinks the tracked git hooks into .git/hooks/.
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
ln -sf "$repo_root/scripts/hooks/pre-commit" "$repo_root/.git/hooks/pre-commit"
chmod +x "$repo_root/scripts/hooks/pre-commit"

echo "Installed pre-commit hook -> .git/hooks/pre-commit"
