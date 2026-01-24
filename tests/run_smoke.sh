#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_PATH=${1:-"$ROOT_DIR/config.llm.json"}
ARTIFACTS_DIR="$ROOT_DIR/tests/_artifacts"
FIXTURES_DIR="$ROOT_DIR/tests/fixtures"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

mkdir -p "$ARTIFACTS_DIR"

CORPUS_MD="$FIXTURES_DIR/corpus.md"
INPUT_MD="$FIXTURES_DIR/input.md"
CORPUS_ZIP="$ARTIFACTS_DIR/corpus.zip"
FINGERPRINT_JSON="$ARTIFACTS_DIR/fingerprint.json"
OUTPUT_MD="$ARTIFACTS_DIR/input.styled.md"

python - <<PY
import zipfile
from pathlib import Path
corpus_md = Path("$CORPUS_MD")
zip_path = Path("$CORPUS_ZIP")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(corpus_md, arcname=corpus_md.name)
PY

python "$ROOT_DIR/fingerprint_style.py" \
  -a "$CORPUS_ZIP" \
  -o "$FINGERPRINT_JSON" \
  -c "$CONFIG_PATH" \
  --max-files 1 \
  --max-bytes-per-file 50000 \
  --excerpt-char-budget 3000

python "$ROOT_DIR/apply_fingerprint.py" \
  -f "$FINGERPRINT_JSON" \
  -i "$INPUT_MD" \
  -o "$OUTPUT_MD" \
  -c "$CONFIG_PATH"

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
