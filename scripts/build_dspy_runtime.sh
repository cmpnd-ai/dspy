#!/usr/bin/env bash
# Build the dspy-runtime distribution from the same source tree as dspy.
#
# Swaps `pyproject-runtime.toml` into place as `pyproject.toml` for the duration
# of the build, then restores the original. Outputs land in `dist/`.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f pyproject-runtime.toml ]]; then
    echo "error: pyproject-runtime.toml not found at $ROOT_DIR" >&2
    exit 1
fi

BACKUP_DIR="$(mktemp -d)"
trap 'mv -f "$BACKUP_DIR/pyproject.toml" pyproject.toml 2>/dev/null || true; rm -rf "$BACKUP_DIR"' EXIT

cp pyproject.toml "$BACKUP_DIR/pyproject.toml"
cp pyproject-runtime.toml pyproject.toml

"${PYTHON:-python3}" -m build "$@"
