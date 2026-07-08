# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## What this is

ATHENA-R1 is an AI agent for treatment reasoning that runs multi-step tool use
over a universe of biomedical tools. The Python package `athena_r1` wraps an
internal reasoning engine in a clean public API; `web/` ships two HTTP servers
around it. See `README.md` for the user-facing overview and `docs/` for eval
details.

## Environment & setup

- **In a Claude Code web session**, the `SessionStart` hook
  (`.claude/hooks/session-start.sh`) builds an isolated `.venv/` and installs
  the lint + offline-test toolchain. `python`, `pytest`, `ruff`, and `athena-r1`
  resolve to that venv. If a command reports missing packages, re-run the hook:
  `CLAUDE_CODE_REMOTE=true ./.claude/hooks/session-start.sh`.
- **Do not install into the base image's system Python.** `tooluniverse` upgrades
  PyJWT, which pip cannot uninstall there (Debian-managed, no RECORD file). Use
  the venv.
- `torch`/`vllm` are **not** installed in web sessions and are not needed for
  lint or the offline tests — the container has no GPU, model weights, or live
  servers. They stay declared in `pyproject.toml` for real inference
  deployments (`pip install -e ".[vllm]"`).

## Commands

All via the `Makefile` (which sets `PYTHONPATH=src`):

| Command | Purpose |
|---|---|
| `make lint` | `ruff check .` |
| `make fmt` | `ruff format` + `ruff check --fix` |
| `make fmt-check` | `ruff format --check` (CI gate) |
| `make test-offline` | Tests that need no live servers — run this locally |
| `make test-full` | Whole suite; integration tests need live vLLM + ToolUniverse |
| `make smoke` | `athena-r1 info` against current env |

CI (`.github/workflows/test.yml`) runs, and PRs must pass:
`ruff check src/ tests/ web/ examples/`, `ruff format --check` on the same dirs,
an import-smoke on Python 3.10/3.11/3.12, and the offline pytest subset. There
is no live vLLM/ToolUniverse in CI, so integration tests auto-skip (see
`tests/conftest.py`).

## Architecture

Runtime needs two backing services: a **vLLM** server hosting the model
(`:8000`) and a **ToolUniverse** HTTP server hosting the tools (`:8080`).
`scripts/launch_*.sh` bring them up.

- `src/athena_r1/agent.py` — **public API**. `AthenaR1`, `AnswerResult`,
  `RoundEvent`, `Backend`. A typed, thin wrapper. Start here.
- `src/athena_r1/_core.py` — the ~2.1k-LoC reasoning engine (`AthenaCore`).
  **Legacy and untyped: excluded from ruff (`extend-exclude`) and from mypy.**
  Don't reformat or re-lint it wholesale; make surgical edits only.
- `src/athena_r1/_prompts.py` — Stage-1/GPT-planning/summary system prompts.
- `src/athena_r1/tool_processors.py` — dispatch for `<tool_call>` blocks
  (`Tool_RAG`, `CallAgent`, `Finish`, `DirectResponse`, `RequireClarification`,
  and `DefaultToolProcessor` for real ToolUniverse tools).
- `src/athena_r1/_report.py` — pure (no-network) helpers that turn a run trace
  into a structured clinical report.
- `src/athena_r1/__main__.py` — the `athena-r1` CLI.
- `web/agui_server.py` — AG-UI protocol server (`:8090`, bundled demo at `/`).
- `web/openai_server.py` — OpenAI-compatible API (`:9000`).

**Two-stage eval protocol:** `agent.answer()` produces free-form reasoning;
`agent.map_to_option()` separately maps a completed conversation to an MCQ
letter. Keep them separate — mixing MCQ prompts into Stage-1 contaminates the
reasoning trace.

## Conventions & gotchas

- Only `agent.py`, `_version.py`, `__main__.py` are type-checked by mypy. New
  public API code should be typed to match.
- The version string lives only in `src/athena_r1/_version.py`.
- `httpx` is used by `_core.py` but not declared in `pyproject.toml`; it arrives
  transitively via `openai`. If you touch dependencies, keep that in mind.
- Relevant env vars: `VLLM_URL`, `TOOLUNIVERSE_API`, `ATHENA_MODEL_PATH`,
  `AZURE_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `ATHENA_R1_LOG_LEVEL`
  (`DEBUG` prints the full trace), `INFER_PRESENCE_PENALTY`.
- Run `make fmt` before committing; CI fails on unformatted code.
