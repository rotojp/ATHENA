#!/bin/bash
# SessionStart hook for ATHENA-R1.
#
# Prepares a fresh Claude Code on the web container so an agent can immediately
# run the linter and the offline test suite (the same checks CI runs). It builds
# an isolated virtualenv and installs the package plus its dev/web tooling into
# it, then exports PATH/PYTHONPATH for the rest of the session.
#
# Why a venv instead of the system interpreter:
#   `tooluniverse` upgrades PyJWT, but the base image ships a Debian-managed
#   PyJWT that pip cannot uninstall (no RECORD file), so a system install aborts.
#   An isolated venv sidesteps every such conflict and is fully reproducible.
#
# Why the offline subset (no torch / vllm):
#   A web container has no GPU, model weights, or live vLLM/ToolUniverse servers,
#   so it can only exercise the offline tests + lint. torch and vllm add hundreds
#   of MB and minutes of install for code paths a web session cannot run. They
#   remain declared in pyproject.toml for real inference deployments; install
#   `pip install -e ".[vllm]"` there.
set -euo pipefail

# Only do the heavy setup in the remote web environment. Local Claude Code users
# keep whatever Python setup they already have.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="$PROJECT_DIR/.venv"
PY="$(command -v python3 || command -v python)"

echo "[session-start] preparing ATHENA-R1 dev environment in $VENV"

# Idempotent: `python -m venv` is safe to re-run, and pip skips already-satisfied
# packages, so warm (cached) containers reach this state almost instantly.
if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV"
fi

# Runtime deps needed to import `athena_r1` + the web servers, plus the lint/test
# tooling. torch/numpy are intentionally omitted (see header); numpy still arrives
# transitively via tooluniverse. Installed before the package so `--no-deps -e .`
# below only registers athena-r1 itself and its console script.
"$VENV/bin/pip" install --quiet --disable-pip-version-check \
  httpx jinja2 "openai>=1.0" "transformers>=4.40" "tooluniverse>=0.3" \
  "fastapi>=0.100" "uvicorn[standard]>=0.20" "ag-ui-protocol>=0.1" \
  "pytest>=8.0" "pytest-timeout>=2.0" "ruff>=0.7" "mypy>=1.10" "requests>=2.31"

"$VENV/bin/pip" install --quiet --disable-pip-version-check --no-deps -e "$PROJECT_DIR"

# Make the venv the default interpreter for this session and mirror the Makefile's
# PYTHONPATH so `python`, `pytest`, `ruff`, and `athena-r1` all resolve here.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export PATH=\"$VENV/bin:\$PATH\""
    echo "export PYTHONPATH=\"$PROJECT_DIR/src\""
  } >> "$CLAUDE_ENV_FILE"
fi

echo "[session-start] ready — run 'make lint' and 'make test-offline'"
