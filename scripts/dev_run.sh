#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"
BASE_URL="http://${HOST}:${PORT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_RETRIES="${DEV_READY_RETRIES:-40}"
SLEEP_S="${DEV_READY_SLEEP_S:-0.25}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting 2api in deterministic local mode..."
MODE=local USE_STUB_ADAPTERS=true HOST="${HOST}" PORT="${PORT}" \
  "${PYTHON_BIN}" -m uvicorn src.server:app --host "${BIND_HOST}" --port "${PORT}" &
SERVER_PID=$!

echo "Waiting for readiness at ${BASE_URL}/ready"
for i in $(seq 1 "${MAX_RETRIES}"); do
  if curl -fsS "${BASE_URL}/ready" >/dev/null 2>&1; then
    echo "✅ Server ready: ${BASE_URL}"
    echo "Next:"
    echo "  1) make smoke"
    echo "  2) make smoke-journey"
    echo "  3) curl ${BASE_URL}/health"
    wait "${SERVER_PID}"
    exit 0
  fi
  echo "  - retry ${i}/${MAX_RETRIES} ..."
  sleep "${SLEEP_S}"
done

echo "❌ Server did not become ready after ${MAX_RETRIES} retries."
exit 1
