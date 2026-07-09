#!/bin/bash
# Launch an MLX server (Apple Silicon / Metal) that serves ATHENA-R1 as a
# drop-in replacement for vLLM on macOS, where vLLM has no build.
#
# Why this works without touching ATHENA's code:
#   ATHENA couples the request `model=` field to the tokenizer id — it does
#   `AutoTokenizer.from_pretrained(model)` — so the server MUST answer to the
#   exact HF id `mims-harvard/ATHENA-R1-Qwen3-8B`. This script serves under that
#   id, so the client's `model=` matches the loaded model and no reload/patch is
#   needed. mlx_lm.server exposes the OpenAI-compatible /v1/completions and
#   /v1/models endpoints ATHENA's client calls.
#
# Usage:
#   bash scripts/launch_mlx.sh [port] [model]
#
# Env knobs:
#   MLX_HOST        bind host (default 127.0.0.1)
#   MLX_VENV        isolated venv path (default <repo>/.venv-mlx)
#   MLX_QUANT_BITS  if set (e.g. 4 or 8), serve a locally-quantized copy to cut
#                   RAM (~16GB bf16 -> ~4.5GB at 4-bit). First run converts.
#   MLX_MODELS_DIR  where quantized copies live (default <repo>/.mlx-models)
set -eu

PORT=${1:-8000}
MODEL=${2:-mims-harvard/ATHENA-R1-Qwen3-8B}
HOST=${MLX_HOST:-127.0.0.1}

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV=${MLX_VENV:-$REPO/.venv-mlx}
PY="$VENV/bin/python"

# MLX is Apple Silicon only.
if [ "$(uname -m)" != "arm64" ]; then
    echo "ERROR: MLX requires Apple Silicon (arm64); this host is $(uname -m)." >&2
    exit 1
fi

# ── isolated venv ────────────────────────────────────────────────────────────
# Kept separate from the project venv on purpose: mlx-lm 0.31.3 breaks at import
# against transformers >=5.11 (a model-registration API change), so transformers
# is pinned to 5.10.0 here without disturbing the dev/offline toolchain.
if [ ! -x "$PY" ]; then
    echo "→ Creating MLX venv at $VENV"
    python3 -m venv "$VENV"
fi
if ! "$PY" -c "import mlx_lm.server" >/dev/null 2>&1; then
    echo "→ Installing mlx-lm (transformers pinned to 5.10.0)…"
    "$VENV/bin/pip" install --quiet --disable-pip-version-check \
        "mlx-lm==0.31.3" "transformers==5.10.0"
fi

# ── optional quantization ────────────────────────────────────────────────────
# The quantized copy MUST live in a directory whose name equals $MODEL, and the
# server must be started from its parent, so that a request for the HF id
# resolves to the local (quantized) weights instead of re-downloading bf16.
SERVE_MODEL="$MODEL"
SERVE_CWD="$REPO"
if [ -n "${MLX_QUANT_BITS:-}" ]; then
    MODELS_DIR=${MLX_MODELS_DIR:-$REPO/.mlx-models}
    QDIR="$MODELS_DIR/$MODEL"
    if [ ! -d "$QDIR" ]; then
        echo "→ Converting $MODEL to ${MLX_QUANT_BITS}-bit MLX at $QDIR (first run only)…"
        "$PY" -m mlx_lm convert --hf-path "$MODEL" --mlx-path "$QDIR" \
            -q --q-bits "$MLX_QUANT_BITS"
    fi
    SERVE_CWD="$MODELS_DIR"   # so the relative name "$MODEL" resolves to $QDIR
    echo "→ Serving ${MLX_QUANT_BITS}-bit copy from $QDIR"
fi

if pgrep -f "mlx_lm server .*--port $PORT" >/dev/null 2>&1; then
    echo "MLX server already running on :$PORT"
    exit 0
fi

echo "Launching MLX server on ${HOST}:$PORT  (model: $MODEL)…"
cd "$SERVE_CWD"
nohup "$PY" -m mlx_lm server --model "$SERVE_MODEL" \
    --host "$HOST" --port "$PORT" \
    > "$REPO/mlx_${PORT}.log" 2>&1 &
echo "  PID=$!  log=mlx_${PORT}.log"

# First run downloads ~16GB from HuggingFace; allow a long warmup.
echo "  waiting for /v1/models (first run downloads weights — can take a while)…"
for i in $(seq 1 240); do
    sleep 5
    if curl -sf "http://${HOST}:${PORT}/v1/models" -m 3 >/dev/null 2>&1; then
        echo "  ✓ MLX ready after $((i*5))s"
        echo
        echo "Point ATHENA at it:"
        echo "  export VLLM_URL=http://${HOST}:${PORT}/v1"
        echo "  export ATHENA_MODEL_PATH=$MODEL"
        echo "  export TOOLUNIVERSE_API=http://127.0.0.1:8080"
        exit 0
    fi
    if ! pgrep -f "mlx_lm server .*--port $PORT" >/dev/null 2>&1; then
        echo "  ✗ MLX server exited — check $REPO/mlx_${PORT}.log" >&2
        tail -8 "$REPO/mlx_${PORT}.log" >&2
        exit 1
    fi
done
echo "ERROR: MLX server did not become healthy in time" >&2
exit 1
