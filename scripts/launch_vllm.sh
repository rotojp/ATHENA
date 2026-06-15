#!/bin/bash
# Launch vLLM to serve ATHENA-R1 on PORT (default 8000).
#
# Usage:
#   bash scripts/launch_vllm.sh [port] [model]

set -eu
PORT=${1:-8000}
MODEL=${2:-mims-harvard/ATHENA-R1-Qwen3-8B}
GPU_MEM=${GPU_MEM:-0.80}

# Redirect vLLM's torch.compile cache to LOCAL node disk. On many shared
# clusters the default (~/.cache/vllm) — or an already-exported VLLM_CACHE_ROOT —
# lives on a quota-limited network filesystem; if that quota is full the
# torch.compile cache write fails AFTER weights load and surfaces as a
# misleading "Engine core initialization failed" (a disk-quota OSError, not
# OOM). We force a local path unless the caller sets ATHENA_VLLM_CACHE to an
# explicit local directory. XDG_CACHE_HOME is also pointed local so
# torch/inductor sub-caches don't fall back to the shared filesystem.
ATHENA_VLLM_CACHE=${ATHENA_VLLM_CACHE:-/tmp/vllm_cache_${USER}}
case "$ATHENA_VLLM_CACHE" in
    # Reject obviously-shared/network locations; fall back to local disk.
    /tmp/*|/dev/shm/*|/scratch/*) : ;;
    /*/netscratch/*|*/.cache/*) ATHENA_VLLM_CACHE=/tmp/vllm_cache_${USER} ;;
esac
export VLLM_CACHE_ROOT="$ATHENA_VLLM_CACHE"
export XDG_CACHE_HOME="${ATHENA_VLLM_CACHE}/xdg"
mkdir -p "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"

nohup vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    > "vllm_${PORT}.log" 2>&1 &
echo "vLLM PID=$!  log=vllm_${PORT}.log  cache=$VLLM_CACHE_ROOT"

# Wait for /v1/models to respond.
for i in $(seq 1 60); do
    sleep 5
    if curl -sf "http://0.0.0.0:${PORT}/v1/models" -m 3 >/dev/null 2>&1; then
        echo "vLLM ready after $((i*5))s"
        exit 0
    fi
done

echo "ERROR: vLLM did not become healthy within 5 minutes" >&2
exit 1
