#!/bin/bash
# Launch the ToolUniverse HTTP server on PORT (default 8080) and wait for
# it to be healthy before returning.
#
# Usage: bash scripts/launch_tooluniverse.sh [port]

set -eu
PORT=${1:-8080}
PYTHON=${PYTHON:-python}

if pgrep -f "http_api_server_cli --host 0.0.0.0 --port $PORT" >/dev/null 2>&1; then
    echo "ToolUniverse server already running on :$PORT"
    exit 0
fi

echo "Launching ToolUniverse server on :$PORT ..."
nohup "$PYTHON" -m tooluniverse.http_api_server_cli \
    --host 0.0.0.0 --port "$PORT" --thread-pool-size 50 \
    > "tooluniverse_${PORT}.log" 2>&1 &
PID=$!
echo "  PID=$PID  log=tooluniverse_${PORT}.log"

# Wait up to 60s for /health.
for i in $(seq 1 30); do
    sleep 2
    if curl -sf "http://0.0.0.0:${PORT}/health" -m 3 >/dev/null 2>&1; then
        echo "  /health OK after ${i}x2s"
        exit 0
    fi
done

echo "ERROR: ToolUniverse did not become healthy within 60s" >&2
exit 1
