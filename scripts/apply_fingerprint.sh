#!/usr/bin/env bash
# Licensed under the PolyForm Noncommercial License 1.0.0.
# Copyright (c) 2026 Nicolas Pepin (npepin@umiquity.com).
# See LICENSE.md for full license text and terms.
# Wrapper script for apply_fingerprint.py.
# - Passes all CLI arguments through unchanged.
# - Keeps invocation consistent across environments.
# - Resolves the repo root relative to this script.
set -euo pipefail

# Resolve symlinks so this script can be called from /usr/local/bin or the repo.
SOURCE_PATH="${BASH_SOURCE[0]}"
while [ -h "$SOURCE_PATH" ]; do
  BASE_DIR="$(cd "$(dirname "$SOURCE_PATH")" && pwd)"
  TARGET="$(readlink "$SOURCE_PATH")"
  if [[ "$TARGET" == /* ]]; then
    SOURCE_PATH="$TARGET"
  else
    SOURCE_PATH="$BASE_DIR/$TARGET"
  fi
done
SCRIPT_DIR="$(cd "$(dirname "$SOURCE_PATH")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Delegate to the Python entry point.
exec python3 "${REPO_ROOT}/apply_fingerprint.py" "$@"
