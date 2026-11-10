#!/usr/bin/env bash
# install-update.sh - rebuild (only when sources changed), launch, and
# health-check the stylometric-mcp server.
#
# The server exposes no /health route; health is verified over JSON-RPC
# (initialize, ping, tools/list) plus one LLM-free `rate` call so the
# measurement/scoring path is exercised end to end.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BIN="target/release/stylometric-mcp"
STAMP="target/.build-stamp"
HOST="${STYLO_MCP_HOST:-127.0.0.1}"
# 8791 matches db-priv/stylo-mcp.sh; 8765 is the fingerprint_api.py default.
PORT="${STYLO_MCP_PORT:-8791}"
CONFIG="${STYLO_MCP_CONFIG:-}"
READY_TIMEOUT="${STYLO_MCP_READY_TIMEOUT:-20}"
RPC_TIMEOUT="${STYLO_MCP_RPC_TIMEOUT:-20}"
FORCE=0
KEEP_RUNNING=0
QUIET=0
SERVER_PID=""
SERVER_LOG=""
STARTED=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Rebuild the Rust MCP server when sources changed, start it, verify health,
then shut it down (unless --keep-running).

Options:
  -f, --force             rebuild even when sources are unchanged
  -c, --config PATH       LLM config file (default: ./config.llm.json, then
                          ../config.llm.json, then the embedded config)
      --host ADDR         bind address (default: ${HOST})
  -p, --port N            bind port (default: ${PORT}; 8765 is used by
                          fingerprint_api.py)
  -k, --keep-running      leave the server running and print its PID
  -q, --quiet             only print warnings and errors
      --offline           force cargo --offline (default: try offline first)
      --online            never pass cargo --offline
  -h, --help              show this help

Environment: STYLO_MCP_HOST, STYLO_MCP_PORT, STYLO_MCP_CONFIG,
             STYLO_MCP_READY_TIMEOUT, STYLO_MCP_RPC_TIMEOUT
EOF
}

log() { [ "${QUIET}" = 1 ] || printf '%s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

CARGO_MODE="auto"
while [ "$#" -gt 0 ]; do
  case "$1" in
    -f|--force) FORCE=1 ;;
    -k|--keep-running) KEEP_RUNNING=1 ;;
    -q|--quiet) QUIET=1 ;;
    --offline) CARGO_MODE="offline" ;;
    --online) CARGO_MODE="online" ;;
    -h|--help) usage; exit 0 ;;
    -c|--config)
      [ "$#" -ge 2 ] || die "$1 requires a value"
      CONFIG="$2"; shift ;;
    --config=*) CONFIG="${1#*=}" ;;
    --host)
      [ "$#" -ge 2 ] || die "$1 requires a value"
      HOST="$2"; shift ;;
    --host=*) HOST="${1#*=}" ;;
    -p|--port)
      [ "$#" -ge 2 ] || die "$1 requires a value"
      PORT="$2"; shift ;;
    --port=*) PORT="${1#*=}" ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
  shift
done

[[ "${PORT}" =~ ^[0-9]+$ ]] || die "invalid port: ${PORT}"
command -v cargo >/dev/null 2>&1 || die "cargo not found in PATH"
command -v curl  >/dev/null 2>&1 || die "curl not found in PATH"
command -v jq    >/dev/null 2>&1 || die "jq not found in PATH"

BASE_URL="http://${HOST}:${PORT}/"

rpc() {
  local payload="$1"
  curl -sS -m "${RPC_TIMEOUT}" -X POST "${BASE_URL}" \
    -H 'Content-Type: application/json' \
    --data-binary "${payload}"
}

rpc_method() {
  local method="$1"
  local params="${2:-{}}"
  rpc "$(jq -nc --arg m "${method}" --argjson p "${params}" \
    '{jsonrpc:"2.0", id:1, method:$m, params:$p}')"
}

compute_source_hash() {
  {
    printf 'schema=1\n'
    cat Cargo.toml
    if [ -f Cargo.lock ]; then cat Cargo.lock; fi
    for f in src/*.rs; do printf '== %s\n' "${f}"; cat "${f}"; done
    for f in resources/*; do printf '== %s\n' "${f}"; cat "${f}"; done
  } | sha256sum | cut -d' ' -f1
}

resolve_config() {
  if [ -n "${CONFIG}" ]; then
    [ -f "${CONFIG}" ] || die "config not found: ${CONFIG}"
    return 0
  fi
  if [ -f "${SCRIPT_DIR}/config.llm.json" ]; then
    CONFIG="${SCRIPT_DIR}/config.llm.json"
  elif [ -f "$(cd "${SCRIPT_DIR}/.." && pwd)/config.llm.json" ]; then
    CONFIG="$(cd "${SCRIPT_DIR}/.." && pwd)/config.llm.json"
  fi
}

probe() {
  rpc_method initialize 2>/dev/null \
    | jq -e '.result.serverInfo.name == "stylometric-mcp"' >/dev/null 2>&1
}

stop_server() {
  if [ "${STARTED}" = 1 ] && [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  STARTED=0
  SERVER_PID=""
  if [ -n "${SERVER_LOG}" ] && [ -f "${SERVER_LOG}" ]; then
    rm -f "${SERVER_LOG}"
  fi
}

trap 'stop_server' EXIT

# --- 1. build when needed ----------------------------------------------------
HASH="$(compute_source_hash)"
REASON=""
if [ "${FORCE}" = 1 ]; then
  REASON="forced"
elif [ ! -x "${BIN}" ]; then
  REASON="binary missing"
elif [ ! -f "${STAMP}" ]; then
  REASON="no build stamp"
elif [ "$(cat "${STAMP}" 2>/dev/null || true)" != "${HASH}" ]; then
  REASON="sources changed"
fi

if [ -n "${REASON}" ]; then
  log "Building (${REASON})..."
  BUILD_OK=0
  case "${CARGO_MODE}" in
    offline) cargo build --release --offline && BUILD_OK=1 ;;
    online)  cargo build --release && BUILD_OK=1 ;;
    auto)
      if cargo build --release --offline; then
        BUILD_OK=1
      else
        log "Offline build failed; retrying with network access..."
        cargo build --release && BUILD_OK=1
      fi
      ;;
  esac
  [ "${BUILD_OK}" = 1 ] || die "cargo build failed"
  [ -x "${BIN}" ] || die "build succeeded but ${BIN} is missing"
  mkdir -p "$(dirname "${STAMP}")"
  printf '%s' "${HASH}" > "${STAMP}"
else
  log "Binary up to date (no rebuild needed)."
fi

# --- 2. verify the binary itself --------------------------------------------
log "Binary: $(ls -l "${BIN}" | awk '{print $5}') bytes"
VERSION="$("${BIN}" --version)" || die "${BIN} --version failed"
[[ "${VERSION}" == stylometric-mcp\ * ]] || die "unexpected version output: ${VERSION}"
log "Version: ${VERSION}"

# --- 3. start the server -----------------------------------------------------
resolve_config
if [ -n "${CONFIG}" ]; then
  log "Config: ${CONFIG}"
else
  log "Warning: no config.llm.json found; the embedded config has a blank api_key."
fi

if probe; then
  log "A server is already listening on ${BASE_URL}; reusing it."
else
  SERVER_LOG="$(mktemp)"
  if [ -n "${CONFIG}" ]; then
    "${BIN}" http --bind "${HOST}:${PORT}" --config "${CONFIG}" >"${SERVER_LOG}" 2>&1 &
  else
    "${BIN}" http --bind "${HOST}:${PORT}" >"${SERVER_LOG}" 2>&1 &
  fi
  SERVER_PID=$!
  STARTED=1
  log "Starting server on ${BASE_URL} (pid ${SERVER_PID})..."
  READY=0
  for _ in $(seq 1 $((READY_TIMEOUT * 10))); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      printf '%s\n' "--- server log ---" >&2
      cat "${SERVER_LOG}" >&2 || true
      die "server exited during startup"
    fi
    if probe; then READY=1; break; fi
    sleep 0.1
  done
  [ "${READY}" = 1 ] || die "server did not become ready within ${READY_TIMEOUT}s"
fi

# --- 4. health checks --------------------------------------------------------
log "Health checks:"

INIT="$(rpc_method initialize)"
echo "${INIT}" | jq -e '.result.serverInfo.name == "stylometric-mcp"' >/dev/null \
  || die "initialize: unexpected server name"
echo "${INIT}" | jq -e '.result.protocolVersion == "2024-11-05"' >/dev/null \
  || die "initialize: unexpected protocol version"
log "  initialize      ok (protocol 2024-11-05)"

PING="$(rpc_method ping)"
echo "${PING}" | jq -e 'has("result") and (has("error") | not)' >/dev/null \
  || die "ping: failed"
log "  ping            ok"

TOOLS="$(rpc_method tools/list)"
echo "${TOOLS}" | jq -e '.result.tools | length == 5' >/dev/null \
  || die "tools/list: expected 5 tools"
for expected in fingerprint apply rate similarity get_fingerprint; do
  echo "${TOOLS}" | jq -e --arg n "${expected}" '.result.tools | any(.name == $n)' >/dev/null \
    || die "tools/list: missing tool ${expected}"
done
log "  tools/list      ok (5 tools)"

RATE_PAYLOAD="$(jq -nc '{jsonrpc:"2.0", id:4, method:"tools/call", params:{name:"rate", arguments:{text:"Small tools work well. They stay clear.", fingerprint:{measurements:{}, targets:{}, validators:{}}}}}')"
RATE="$(rpc "${RATE_PAYLOAD}")"
echo "${RATE}" | jq -e '.result.isError == false' >/dev/null \
  || die "rate: tool call failed: $(echo "${RATE}" | jq -r '.result.content[0].text // .error.message // "unknown"')"
echo "${RATE}" | jq -e '.result.structuredContent.compliance.score | type == "number"' >/dev/null \
  || die "rate: compliance score missing"
SCORE="$(echo "${RATE}" | jq -r '.result.structuredContent.compliance.score')"
log "  tools/call rate ok (score ${SCORE})"

UNKNOWN="$(rpc_method no_such_method || true)"
echo "${UNKNOWN}" | jq -e '.error.code == -32601' >/dev/null \
  || die "unknown method: expected JSON-RPC -32601"
log "  error handling  ok (-32601 for unknown method)"

# --- 5. finish ---------------------------------------------------------------
if [ "${KEEP_RUNNING}" = 1 ]; then
  trap - EXIT
  if [ "${STARTED}" = 1 ]; then
    log ""
    log "Server running: ${BASE_URL} (pid ${SERVER_PID})"
    log "Stop it with: kill ${SERVER_PID}"
    printf '%s\n' "${SERVER_PID}"
  else
    log ""
    log "Server already running on ${BASE_URL} (not started by this script)."
  fi
else
  if [ "${STARTED}" = 1 ]; then
    stop_server
    log ""
    log "All health checks passed; server stopped."
  else
    log ""
    log "All health checks passed (server was already running; left untouched)."
  fi
fi
