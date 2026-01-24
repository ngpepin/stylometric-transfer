#!/usr/bin/env bash
# Licensed under the PolyForm Noncommercial License 1.0.0.
# Copyright (c) 2026 Nicolas Pepin (npepin@umiquity.com).
# See LICENSE.md for full license text and terms.
# Wrapper script for fingerprint_style.py.
# - Passes all CLI arguments through unchanged.
# - Keeps invocation consistent across environments.
# - Resolves the repo root relative to this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Delegate to the Python entry point.
exec python3 "${REPO_ROOT}/fingerprint_style.py" "$@"
