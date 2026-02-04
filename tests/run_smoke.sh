#!/usr/bin/env bash
# Licensed under the PolyForm Noncommercial License 1.0.0.
# Copyright (c) 2026 Nicolas Pepin (npepin@umiquity.com).
# See LICENSE.md for full license text and terms.
# End-to-end smoke test for the pipeline.
# - Builds a tiny corpus zip from fixtures
# - Runs fingerprinting and apply steps
# - Validates that outputs are present and parseable
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Allow overriding the config path (defaults to repo root config.llm.json).
CONFIG_PATH=${1:-"$ROOT_DIR/config.llm.json"}
ARTIFACTS_DIR="$ROOT_DIR/tests/_artifacts"
FIXTURES_DIR="$ROOT_DIR/tests/fixtures"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

# Ensure artifacts directory exists.
mkdir -p "$ARTIFACTS_DIR"

# Run regression tests (no API calls).
"$ROOT_DIR/tests/run_v1_1_0_regression.sh"
"$ROOT_DIR/tests/run_v1_5_X_regression.sh"
"$ROOT_DIR/tests/run_v1_7_X_regression.sh"

# LLM connectivity smoke test (requires API config).
LLM_SMOKE_CONFIG="$CONFIG_PATH" python "$ROOT_DIR/tests/test_llm_smoke.py"

CORPUS_MD="$FIXTURES_DIR/corpus.md"
INPUT_MD="$FIXTURES_DIR/input.md"
CORPUS_ZIP="$ARTIFACTS_DIR/corpus.zip"
FINGERPRINT_JSON="$ARTIFACTS_DIR/fingerprint.json"
OUTPUT_MD="$ARTIFACTS_DIR/input.styled.md"

# Create a tiny corpus archive from the fixture.
python - <<PY
import zipfile
from pathlib import Path
corpus_md = Path("$CORPUS_MD")
zip_path = Path("$CORPUS_ZIP")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(corpus_md, arcname=corpus_md.name)
PY

# Run fingerprinting using the minimal corpus.
python "$ROOT_DIR/fingerprint_style.py" \
  -a "$CORPUS_ZIP" \
  -o "$FINGERPRINT_JSON" \
  -c "$CONFIG_PATH" \
  --max-files 1 \
  --max-bytes-per-file 50000 \
  --excerpt-char-budget 3000

# Apply the fingerprint to the input fixture.
python "$ROOT_DIR/apply_fingerprint.py" \
  -f "$FINGERPRINT_JSON" \
  -i "$INPUT_MD" \
  -o "$OUTPUT_MD" \
  -c "$CONFIG_PATH"

# Sanity-check outputs.
python - <<PY
import json
from pathlib import Path
fp = Path("$FINGERPRINT_JSON")
if not fp.exists():
    raise SystemExit("fingerprint.json missing")
json.loads(fp.read_text(encoding="utf-8"))

out_md = Path("$OUTPUT_MD")
if not out_md.exists():
    raise SystemExit("styled markdown missing")
if not out_md.read_text(encoding="utf-8").strip():
    raise SystemExit("styled markdown empty")
PY

echo "Smoke test complete. Artifacts in $ARTIFACTS_DIR" 
