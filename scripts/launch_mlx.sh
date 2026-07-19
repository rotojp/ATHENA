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
#   MLX_QUANT_BITS  quantization width. Unset -> auto-select from physical RAM
#                   (the 8B model is ~16GB bf16 / ~8.5GB 8-bit / ~4.5GB 4-bit, so
#                   a RAM-tight Mac otherwise swaps and generation crawls). Set a
#                   number (e.g. 4 or 8) to pin it, or `native`/`0` for full
#                   precision. First quantized run converts once and caches.
#   MLX_MODELS_DIR  where quantized copies live (default <repo>/.mlx-models)
set -eu

PORT=${1:-8000}
MODEL=${2:-mims-harvard/ATHENA-R1-Qwen3-8B}
HOST=${MLX_HOST:-127.0.0.1}

# Tolerate `launch_mlx.sh <model>` (model as the first arg, mirroring
# launch_all.sh). If $1 isn't a numeric port, treat it as the model and keep the
# default port — otherwise a model id like "mims-harvard/ATHENA-R1-Qwen3-8B"
# becomes the port, the bind is nonsense, and the log path mlx_<port>.log gains
# a "/" that breaks the redirect ("No such file or directory").
if ! printf '%s' "$PORT" | grep -qE '^[0-9]+$'; then
    MODEL=$PORT
    PORT=8000
fi

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

# ── choose quantization ──────────────────────────────────────────────────────
# If the caller hasn't pinned MLX_QUANT_BITS, pick a width that fits physical RAM
# (unified memory on Apple Silicon), leaving headroom for the KV cache and the
# separate ToolUniverse embedding server. This is the future-proof default: it
# stops a RAM-tight Mac from silently serving bf16 into swap.
if [ -z "${MLX_QUANT_BITS:-}" ]; then
    RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
    if   [ "$RAM_GB" -ge 32 ]; then MLX_QUANT_BITS=native
    elif [ "$RAM_GB" -ge 20 ]; then MLX_QUANT_BITS=8
    else                            MLX_QUANT_BITS=4
    fi
    echo "→ Detected ${RAM_GB}GB RAM; auto-selected MLX_QUANT_BITS=$MLX_QUANT_BITS (set it to override)"
fi

# ── quantization ─────────────────────────────────────────────────────────────
# For a quantized width, the copy MUST live in a directory whose name equals
# $MODEL, and the server must start from its parent, so a request for the HF id
# resolves to the local (quantized) weights instead of re-downloading bf16.
# `native`/`0` serves the full-precision HF id directly.
SERVE_MODEL="$MODEL"
SERVE_CWD="$REPO"
case "$MLX_QUANT_BITS" in
    native | 0)
        echo "→ Serving $MODEL at full precision (no quantization)"
        ;;
    *)
        MODELS_DIR=${MLX_MODELS_DIR:-$REPO/.mlx-models}
        QDIR="$MODELS_DIR/$MODEL"
        # Treat the cached copy as usable only if BOTH config.json and the
        # weights landed. An interrupted/OOM-killed convert leaves a partial dir;
        # mlx_lm would then serve broken weights and `mlx_lm convert` refuses to
        # overwrite an existing path, so the stack would stay broken across
        # re-runs. Purge a partial dir so this run re-converts cleanly.
        if [ -d "$QDIR" ] && { [ ! -f "$QDIR/config.json" ] || ! ls "$QDIR"/*.safetensors >/dev/null 2>&1; }; then
            echo "→ Removing incomplete quantized copy at $QDIR (will re-convert)"
            rm -rf "$QDIR"
        fi
        if [ ! -d "$QDIR" ]; then
            # mlx_lm convert's save step re-resolves the source repo with
            # local_files_only=True and aborts with IncompleteSnapshotError if the
            # HF cache is missing any file (a running server fetches only the
            # weights it needs, leaving e.g. README.md / .gitattributes absent).
            # Ensure the full snapshot is present first; skip quietly for a local
            # path (snapshot_download raises on a non-repo-id).
            echo "→ Ensuring complete HF snapshot of $MODEL (mlx convert requires it)…"
            "$PY" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "$MODEL" || true
            echo "→ Converting $MODEL to ${MLX_QUANT_BITS}-bit MLX at $QDIR (first run only)…"
            "$PY" -m mlx_lm convert --hf-path "$MODEL" --mlx-path "$QDIR" \
                -q --q-bits "$MLX_QUANT_BITS"
        fi
        SERVE_CWD="$MODELS_DIR"   # so the relative name "$MODEL" resolves to $QDIR
        echo "→ Serving ${MLX_QUANT_BITS}-bit copy from $QDIR"
        ;;
esac

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
