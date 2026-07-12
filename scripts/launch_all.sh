#!/bin/bash
# One-shot, IDEMPOTENT launcher for the full ATHENA-R1 stack:
#   • ToolUniverse HTTP server on :8080
#   • model server on :8000 — vLLM, or MLX on Apple Silicon (vLLM has no macOS
#     build); override with MODEL_BACKEND=vllm|mlx
#   • AG-UI server on :8090 (with the bundled browser demo at /)
#   • OpenAI-compat server on :9000
#
# Safe to re-run: each service that is ALREADY healthy is left alone, so after
# a node rotation you can just re-run this one command and it brings back only
# what's missing. Use --restart to force-kill and relaunch everything.
#
# Usage:
#   bash scripts/launch_all.sh [model_id_or_path]      # bring up what's missing
#   bash scripts/launch_all.sh --restart [model]       # kill + relaunch all
#   bash scripts/launch_all.sh --status                # just report health
#   bash scripts/launch_all.sh --stop                  # stop the stack
#
# Logs land under $LOG_DIR (default /tmp). Tail: tail -f /tmp/athena_*.log

set -u

RESTART=0
ACTION=up
ARGS=()
for a in "$@"; do
    case "$a" in
        --restart) RESTART=1 ;;
        --status)  ACTION=status ;;
        --stop)    ACTION=stop ;;
        *)         ARGS+=("$a") ;;
    esac
done
MODEL=${ARGS[0]:-mims-harvard/ATHENA-R1-Qwen3-8B}
LOG_DIR=${LOG_DIR:-/tmp}
PYTHON=${PYTHON:-python}

cd "$(dirname "$0")/.."

# Prefer the repo venv so `vllm`, the web servers, and tooluniverse all resolve
# to the project's installed interpreter regardless of the caller's active env.
[ -d "$PWD/.venv/bin" ] && PATH="$PWD/.venv/bin:$PATH"
if [ -x "$PWD/.venv/bin/python" ] && { [ -z "${PYTHON:-}" ] || [ "${PYTHON}" = "python" ]; }; then
    PYTHON="$PWD/.venv/bin/python"
fi

# Model-server backend: vLLM has no macOS build, so serve via MLX on Apple
# Silicon. Override with MODEL_BACKEND=vllm|mlx.
if [ -z "${MODEL_BACKEND:-}" ]; then
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        MODEL_BACKEND=mlx
    else
        MODEL_BACKEND=vllm
    fi
fi

# Probe/connect over 127.0.0.1, not 0.0.0.0: the loopback-bound servers (TU and
# MLX default to 127.0.0.1) are unreachable via 0.0.0.0 on macOS, and 127.0.0.1
# also reaches a 0.0.0.0-bound server (vLLM) on Linux — so it's correct on both.
healthy() {  # healthy <port> <path>
    curl -sf "http://127.0.0.1:${1}${2}" -m 3 >/dev/null 2>&1
}
report() {
    healthy 8080 /health        && echo "  ✓ ToolUniverse  :8080" || echo "  ✗ ToolUniverse  :8080"
    healthy 8000 /v1/models     && echo "  ✓ model ($MODEL_BACKEND) :8000" || echo "  ✗ model ($MODEL_BACKEND) :8000"
    healthy 8090 /health        && echo "  ✓ AG-UI demo    :8090" || echo "  ✗ AG-UI demo    :8090"
    healthy 9000 /health        && echo "  ✓ OpenAI-compat :9000" || echo "  ✗ OpenAI-compat :9000"
}

if [ "$ACTION" = status ]; then
    echo "Stack health on $(hostname -s):"; report; exit 0
fi

stop_stack() {
    echo "→ Stopping stack…"
    pkill -u "$USER" -f "web/agui_server.py"   2>/dev/null || true
    pkill -u "$USER" -f "web/openai_server.py" 2>/dev/null || true
    pkill -u "$USER" -f "vllm serve"           2>/dev/null || true
    pkill -u "$USER" -f "EngineCore"           2>/dev/null || true
    pkill -u "$USER" -f "launch_mlx.sh"        2>/dev/null || true
    pkill -u "$USER" -f "mlx_lm"               2>/dev/null || true
    pkill -u "$USER" -f "tooluniverse.http"    2>/dev/null || true
    sleep 3
}

if [ "$ACTION" = stop ]; then stop_stack; echo "Stopped."; report; exit 0; fi
if [ "$RESTART" = 1 ]; then stop_stack; fi

# ── ToolUniverse ─────────────────────────────────────────────────────────
if healthy 8080 /health; then
    echo "✓ ToolUniverse already up on :8080 (skipping)"
else
    echo "→ Starting ToolUniverse on :8080 (log: ${LOG_DIR}/athena_tu.log)"
    bash scripts/launch_tooluniverse.sh 8080 > "${LOG_DIR}/athena_tu.log" 2>&1 &
    disown
fi

# ── model server (:8000) ───────────────────────────────────────────────────
if healthy 8000 /v1/models; then
    echo "✓ model server already up on :8000 (skipping)"
elif [ "$MODEL_BACKEND" = mlx ]; then
    echo "→ Starting MLX server on :8000 with $MODEL (log: ${LOG_DIR}/athena_mlx.log)"
    bash scripts/launch_mlx.sh 8000 "$MODEL" > "${LOG_DIR}/athena_mlx.log" 2>&1 &
    disown
    echo "  waiting for :8000 /v1/models (first run converts weights — can take a while)…"
    for i in $(seq 1 360); do
        healthy 8000 /v1/models && { echo "  ✓ MLX ready after $((i*5))s"; break; }
        # Fail fast if the launcher died. Match the launcher and both mlx_lm
        # phases (convert, then serve) so the cold conversion isn't mistaken for
        # a crash.
        if ! pgrep -u "$USER" -f "launch_mlx.sh|mlx_lm" >/dev/null 2>&1; then
            echo "  ✗ MLX process exited — check ${LOG_DIR}/athena_mlx.log" >&2
            tail -5 "${LOG_DIR}/athena_mlx.log" 2>/dev/null >&2
            exit 1
        fi
        sleep 5
    done
else
    echo "→ Starting vLLM on :8000 with $MODEL (log: ${LOG_DIR}/athena_vllm.log)"
    bash scripts/launch_vllm.sh 8000 "$MODEL" > "${LOG_DIR}/athena_vllm.log" 2>&1 &
    disown
    echo "  waiting for vLLM /v1/models (cold start can take 10-25 min on a busy node)…"
    for i in $(seq 1 360); do
        healthy 8000 /v1/models && { echo "  ✓ vLLM ready after $((i*5))s"; break; }
        # Fail fast if the process died (e.g. a disk-quota crash on a full cache dir).
        if ! pgrep -u "$USER" -f "vllm serve" >/dev/null 2>&1; then
            echo "  ✗ vLLM process exited — check ${LOG_DIR}/athena_vllm.log" >&2
            grep -iE "Disk quota|OutOfMemory|Error" "${LOG_DIR}/athena_vllm.log" 2>/dev/null | tail -3 >&2
            exit 1
        fi
        sleep 5
    done
fi

export VLLM_URL=http://127.0.0.1:8000/v1
export TOOLUNIVERSE_API=http://127.0.0.1:8080
export ATHENA_MODEL_PATH="$MODEL"
export AZURE_API_KEY=${AZURE_API_KEY:-dummy}
export ATHENA_R1_LOG_LEVEL=${ATHENA_R1_LOG_LEVEL:-WARNING}
export ATHENA_MAX_AGENT_LEVEL=${ATHENA_MAX_AGENT_LEVEL:-1}
export PYTHONUNBUFFERED=1

# ── AG-UI demo ───────────────────────────────────────────────────────────
if healthy 8090 /health; then
    echo "✓ AG-UI demo already up on :8090 (skipping)"
else
    echo "→ Starting AG-UI server on :8090 (log: ${LOG_DIR}/athena_agui.log)"
    PYTHONPATH=src AGUI_PORT=8090 setsid nohup "$PYTHON" -u web/agui_server.py \
        > "${LOG_DIR}/athena_agui.log" 2>&1 &
    disown
fi

# ── OpenAI-compat ────────────────────────────────────────────────────────
if healthy 9000 /health; then
    echo "✓ OpenAI-compat already up on :9000 (skipping)"
else
    echo "→ Starting OpenAI-compat server on :9000 (log: ${LOG_DIR}/athena_openai.log)"
    PYTHONPATH=src PORT=9000 setsid nohup "$PYTHON" -u web/openai_server.py \
        > "${LOG_DIR}/athena_openai.log" 2>&1 &
    disown
fi

# Give the web servers a moment to warm the agent, then report.
echo "  waiting for web servers to bind…"
for i in $(seq 1 60); do
    if healthy 8090 /health && healthy 9000 /health; then break; fi
    sleep 5
done

echo
echo "Stack health on $(hostname -s):"; report
echo
echo "Open the bundled browser demo:"
echo "  http://$(hostname -f):8090/"
