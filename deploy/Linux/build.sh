#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
BUILD_DIR="${ROOT_DIR}/build"

cd "${ROOT_DIR}"

VENV_DIR="${ROOT_DIR}/.venv-build"
PYTHON_BIN="python3"

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"

# Ensure dependencies are available in the venv
"${VENV_PY}" -m pip install --upgrade pip >/dev/null 2>&1 || true
"${VENV_PY}" -m pip install --upgrade -r "${ROOT_DIR}/requirements.txt"

# Ensure PyInstaller is available
"${VENV_PY}" -m PyInstaller --version >/dev/null 2>&1 || {
  echo "PyInstaller is not available after install." >&2
  exit 1
}

# Clean previous builds
rm -rf "${DIST_DIR}" "${BUILD_DIR}"

# Build fingerprint_style executable
"${VENV_PY}" -m PyInstaller \
  --clean \
  --onefile \
  --name fingerprint_style \
  fingerprint_style.py

# Build apply_fingerprint executable
"${VENV_PY}" -m PyInstaller \
  --clean \
  --onefile \
  --name apply_fingerprint \
  apply_fingerprint.py

echo "Build complete. Binaries are in ${DIST_DIR}" 
