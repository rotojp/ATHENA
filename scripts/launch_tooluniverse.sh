#!/bin/bash
# Launch the ToolUniverse HTTP server on PORT (default 8080) and wait for
# it to be healthy before returning.
#
# Usage: bash scripts/launch_tooluniverse.sh [port]

set -eu
PORT=${1:-8080}

# Prefer the repo's own venv interpreter so the script works regardless of the
# caller's active environment (a bare `python` may resolve to a venv without
# tooluniverse installed).
REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$REPO/.venv/bin/python" ]; then PYTHON="$REPO/.venv/bin/python"; else PYTHON=python; fi
fi

# ToolUniverse >=1.x refuses to bind a non-loopback host without an auth token.
# Default to loopback for local dev; override HOST=0.0.0.0 (with
# TOOLUNIVERSE_API_TOKEN set) to expose it to other machines.
HOST=${HOST:-127.0.0.1}

# Tool_RAG (the tool-retrieval step ATHENA calls) needs ToolUniverse's embedding
# stack (sentence-transformers, torch, faiss). Install it once into this
# interpreter's environment if missing; non-fatal so the server still starts
# (Tool_RAG just stays unavailable) if the install can't run. Set
# ATHENA_SKIP_EMBEDDING=1 to skip this check entirely.
if [ "${ATHENA_SKIP_EMBEDDING:-}" != "1" ] && \
   ! "$PYTHON" -c "import sentence_transformers" >/dev/null 2>&1; then
    echo "→ Installing tooluniverse[embedding] for Tool_RAG (one-time)…"
    "$PYTHON" -m pip install --quiet --disable-pip-version-check "tooluniverse[embedding]" \
        || echo "  ⚠ embedding install failed — server will start but Tool_RAG will be unavailable" >&2
fi

if pgrep -f "http_api_server_cli --host $HOST --port $PORT" >/dev/null 2>&1; then
    echo "ToolUniverse server already running on ${HOST}:$PORT"
    exit 0
fi

echo "Launching ToolUniverse server on ${HOST}:$PORT ..."
nohup "$PYTHON" -m tooluniverse.http_api_server_cli \
    --host "$HOST" --port "$PORT" --thread-pool-size 50 \
    > "tooluniverse_${PORT}.log" 2>&1 &
PID=$!
echo "  PID=$PID  log=tooluniverse_${PORT}.log"

# Wait up to 60s for /health.
for i in $(seq 1 30); do
    sleep 2
    if curl -sf "http://${HOST}:${PORT}/health" -m 3 >/dev/null 2>&1; then
        echo "  /health OK after ${i}x2s"
        exit 0
    fi
done

echo "ERROR: ToolUniverse did not become healthy within 60s" >&2
exit 1
