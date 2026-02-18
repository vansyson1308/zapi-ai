#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8011}"
BASE_URL="http://127.0.0.1:${PORT}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "Starting 2api server in deterministic LOCAL mode on port ${PORT}..."
MODE=local USE_STUB_ADAPTERS=true PORT="${PORT}" \
  uvicorn src.server:app --host 127.0.0.1 --port "${PORT}" >/tmp/2api-smoke.log 2>&1 &
SERVER_PID=$!

for _ in {1..40}; do
  if curl -fsS "${BASE_URL}/ready" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

echo "\n1) GET /health"
curl -fsS "${BASE_URL}/health" | python3 -c 'import json,sys; d=json.load(sys.stdin); print({"status":d.get("status"),"mode":d.get("mode")})'

echo "\n2) POST /v1/chat/completions (non-stream)"
curl -fsS "${BASE_URL}/v1/chat/completions" \
  -H "Authorization: Bearer 2api_demo" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","stream":false,"messages":[{"role":"user","content":"hello"}]}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); c=d.get("choices",[{}])[0]; m=c.get("message",{}); print({"object":d.get("object"),"content":m.get("content")})'

echo "\n3) POST /v1/chat/completions (stream)"
STREAM_OUT=$(mktemp)
curl -fsS -N "${BASE_URL}/v1/chat/completions" \
  -H "Authorization: Bearer 2api_demo" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","stream":true,"messages":[{"role":"user","content":"hello"}]}' > "${STREAM_OUT}"

tail -n 5 "${STREAM_OUT}"
if ! grep -q "data: \[DONE\]" "${STREAM_OUT}"; then
  echo "ERROR: stream did not end with [DONE]"
  exit 1
fi

echo "\nSmoke user journey completed successfully."
