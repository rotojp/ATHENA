"""OpenAI-compatible HTTP server wrapping ATHENA-R1.

Lets third-party clients (Open WebUI, LibreChat, any `openai` SDK user) call
ATHENA-R1 as if it were a hosted OpenAI chat model.

Run:
    uvicorn athena_r1.web.openai_server:app --host 0.0.0.0 --port 9000
or:
    python web/openai_server.py

Client usage:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:9000/v1", api_key="any")
    resp = client.chat.completions.create(
        model="athena-r1",
        messages=[{"role": "user", "content": "Dose adjustment for X in CKD?"}],
    )

Pair with Open WebUI for a full chat UI (no frontend code required):
    docker run -d -p 3000:8080 \
        -e OPENAI_API_BASE_URL=http://host.docker.internal:9000/v1 \
        -e OPENAI_API_KEY=anything \
        ghcr.io/open-webui/open-webui:main
"""

from __future__ import annotations

import json
import os

# Local sibling helper — shared with web/agui_server.py so both web entry
# points fold multi-turn history into Stage-1 prompts identically.
import sys as _sys  # noqa: E402
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from athena_r1 import AthenaR1, Backend

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _messages import fold_messages_to_prompt as _fold_messages_to_prompt  # noqa: E402

# ─── Request / response schemas (subset of OpenAI ChatCompletion) ──────────


class ChatMessage(BaseModel):
    role: str
    content: str


class StreamOptions(BaseModel):
    # OpenAI's stream_options.include_usage: when true, the streaming
    # response emits one final chunk (after the finish_reason chunk) whose
    # ``choices`` is empty and which carries a populated ``usage`` object.
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = "athena-r1"
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    report: bool = (
        False  # when true (non-streaming), return a detailed report instead of the answer
    )

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        if v < 0.0 or v > 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {v}")
        return v

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v: float) -> float:
        if v <= 0.0 or v > 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {v}")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _check_max_tokens(cls, v: int | None) -> int | None:
        # OpenAI accepts ``null`` (unset) but rejects non-positive integers.
        # vLLM downstream would either error or behave unpredictably on a
        # negative max_tokens, so reject it at the boundary with a clear 422.
        if v is not None and v <= 0:
            raise ValueError(f"max_tokens must be > 0, got {v}")
        return v


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage = Field(default_factory=Usage)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "mims-harvard"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ─── App + agent singleton ────────────────────────────────────────────────


app = FastAPI(title="ATHENA-R1 OpenAI-compatible server", version="1.0.0")

_agent: AthenaR1 | None = None
_model_id: str = os.environ.get("ATHENA_MODEL_ID", "athena-r1")
# Serializes lazy init so two concurrent /v1/chat/completions requests don't
# both construct + load the model (FastAPI runs sync `def` endpoints in a
# threadpool, so a race is real). Without this, the second caller can see a
# half-built `_agent` (constructed but pre-`init()`) or both can recurse into
# init_model concurrently.
import threading as _threading  # noqa: E402

_agent_lock = _threading.Lock()


def get_agent() -> AthenaR1:
    global _agent
    # ``getattr(..., True)`` lets unit-test stubs without `_initialized`
    # pass through without re-init.
    if _agent is not None and getattr(_agent, "_initialized", True):
        return _agent
    with _agent_lock:
        if _agent is None:
            _agent = AthenaR1(**_agent_kwargs())
        if not getattr(_agent, "_initialized", True):
            _agent.init()
    return _agent


def _agent_kwargs() -> dict:
    """Build AthenaR1 kwargs from env, honoring the selected backend.

    ``ATHENA_BACKEND`` picks athena (local, default) | gpt | claude | gemini;
    ``ATHENA_BACKEND_MODEL`` overrides the backend's model id. Only the local
    backend needs a served model + vLLM endpoint.
    """
    valid = {b.value for b in Backend}
    backend_name = os.environ.get("ATHENA_BACKEND", "athena").strip().lower()
    backend = Backend(backend_name) if backend_name in valid else Backend.ATHENA
    kwargs: dict = {
        "backend": backend,
        "backend_model": os.environ.get("ATHENA_BACKEND_MODEL") or None,
        "tool_server": os.environ.get("TOOLUNIVERSE_API", "http://127.0.0.1:8080"),
    }
    if backend == Backend.ATHENA:
        kwargs["model"] = os.environ.get("ATHENA_MODEL_PATH", "mims-harvard/ATHENA-R1-Qwen3-8B")
        kwargs["vllm_url"] = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1")
    return kwargs


@app.on_event("startup")
def _warm_agent_in_background() -> None:
    """Pre-load the agent (tokenizer + ToolRAG embedding model) in a daemon
    thread at startup so the first /v1/chat/completions request doesn't eat
    the full ~30-60s init cost. ``/health`` stays responsive; a request
    arriving mid-warm blocks on the same ``_agent_lock`` get_agent() uses,
    so there's no double init. Warm failures are swallowed — the first real
    request retries and surfaces any error then.
    """

    def _warm() -> None:
        try:
            get_agent()
        except Exception:  # noqa: BLE001 — best-effort warmup, never fatal
            pass

    _threading.Thread(target=_warm, name="agent-warmup", daemon=True).start()


@app.get("/v1/models", response_model=ModelList)
def list_models() -> ModelList:
    return ModelList(
        data=[ModelInfo(id=_model_id, created=int(time.time()))],
    )


def _count_tokens(agent: AthenaR1, text: str) -> int:
    """Token count via the model tokenizer, with a whitespace fallback.

    The fallback (word count) keeps ``usage`` non-zero even if the tokenizer
    isn't reachable for any reason — better an approximate count than the
    silent 0/0/0 that breaks SDK cost-tracking.
    """
    if not text:
        return 0
    tok = getattr(getattr(agent, "_core", None), "tokenizer", None)
    if tok is not None:
        try:
            return len(tok.encode(text))
        except Exception:  # noqa: BLE001
            pass
    return len(text.split())


def _count_usage(agent: AthenaR1, prompt_text: str, completion_text: str) -> Usage:
    p = _count_tokens(agent, prompt_text)
    c = _count_tokens(agent, completion_text)
    return Usage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


def _extract_user_question(req: ChatCompletionRequest) -> str:
    """Build the Stage-1 prompt from the chat history.

    Delegates to the shared ``web/_messages.fold_messages_to_prompt`` helper
    so the AG-UI and OpenAI-compat servers fold conversation history into
    Stage-1 prompts identically. Raises HTTP 400 if no user message is
    present, preserving the previous behaviour of this function.
    """
    prompt = _fold_messages_to_prompt(req.messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="no user message found")
    return prompt


def _stream_response(req: ChatCompletionRequest, question: str):
    """Generator that emits OpenAI-format SSE chunks while ATHENA-R1 reasons.

    The ``question`` is extracted BEFORE the generator is started so any
    HTTPException raised by validation surfaces as a proper 4xx response,
    not as an abrupt mid-stream connection close after StreamingResponse
    has already flushed its headers.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    final_text = ""  # top-level answer text, captured for include_usage

    def chunk(delta_text: str, finish_reason: str | None = None, role: str | None = None) -> str:
        delta: dict[str, Any] = {}
        if role is not None:
            delta["role"] = role
        if delta_text:
            delta["content"] = delta_text
        payload = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": _model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # Opening chunk carries `role: "assistant"` (matches OpenAI's first-chunk
    # behaviour). Some SDK clients (langchain.OpenAI, older openai-python)
    # rely on the role being set on the first delta to identify the speaker
    # and may skip subsequent content if it isn't.
    yield chunk("", role="assistant")

    try:
        # Keep lazy init inside the try so a first-request failure becomes an
        # in-stream [error] chunk + [DONE], not a malformed stream (headers are
        # already flushed by the time this generator runs).
        agent = get_agent()
        for ev in agent.answer_streaming(
            question,
            temperature=req.temperature,
            top_p=req.top_p,
            max_new_tokens=req.max_tokens or 4096,
        ):
            # Prefix every event with the emitting agent's id when not main.
            scope = f"[{ev.agent_id}] " if ev.agent_id != "main" else ""
            if ev.type == "round_start":
                yield chunk(f"\n{scope}[round {ev.round}]\n")
            elif ev.type == "reasoning":
                yield chunk(f"{scope}{ev.content or ''}\n")
            elif ev.type == "tool_rag_query":
                yield chunk(f"{scope}[tool_rag_query] {ev.content or ''}\n")
            elif ev.type == "tools_retrieved":
                tools = ev.metadata.get("tools", [])
                yield chunk(f"{scope}[tools_retrieved] {', '.join(tools) or ev.content or ''}\n")
            elif ev.type == "tool_call":
                tool_name = ev.metadata.get("tool_name") or "tool"
                # Tool_RAG / CallAgent already emit their own dedicated events.
                if tool_name in ("Tool_RAG", "CallAgent"):
                    continue
                yield chunk(f"{scope}[tool_call {tool_name}] {(ev.content or '')[:200]}\n")
            elif ev.type == "tool_result":
                name = ev.metadata.get("name", "tool")
                snippet = (ev.content or "")[:200]
                yield chunk(f"{scope}[tool_result {name}] {snippet}\n")
            elif ev.type == "subagent_start":
                yield chunk(f"{scope}[subagent_start agent_id={ev.agent_id}] {ev.content or ''}\n")
            elif ev.type == "subagent_end":
                yield chunk(
                    f"{scope}[subagent_end agent_id={ev.agent_id}] {(ev.content or '')[:200]}\n"
                )
            elif ev.type == "final_answer":
                text = ev.content or ""
                if "</think>" in text:
                    text = text.split("</think>", 1)[-1].strip()
                # Only top-level final emits stop; nested ones are just text.
                if ev.agent_level == 0:
                    # `forced=True` means we hit max_round or a timeout and the
                    # engine synthesized an answer from a partial trace —
                    # map to the OpenAI "length" finish_reason so SDK clients
                    # can distinguish a complete answer from a truncated one.
                    fr = "length" if ev.metadata.get("forced") else "stop"
                    final_text = text  # captured for include_usage accounting
                    yield chunk(f"\n[final]\n{text}\n", finish_reason=fr)
                else:
                    yield chunk(f"{scope}[final] {text[:200]}\n")
    except Exception as e:  # noqa: BLE001
        # OpenAI streaming spec only defines finish_reason values
        # {stop,length,tool_calls,content_filter,function_call}. Strict SDK
        # clients (LiteLLM, openai-python ≥1.40) reject unknown enums, so we
        # emit the error text in the delta and close with "stop".
        yield chunk(f"\n[error] {e}\n", finish_reason="stop")

    # OpenAI stream_options.include_usage: emit one final chunk with empty
    # ``choices`` and a populated ``usage`` object, just before [DONE].
    # Mirrors the non-streaming usage accounting (client input → final
    # answer). Counted on the top-level final answer text we captured above.
    if req.stream_options is not None and req.stream_options.include_usage:
        prompt_text = "\n".join(m.content for m in req.messages if getattr(m, "content", None))
        usage = _count_usage(agent, prompt_text, final_text)
        usage_payload = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": _model_id,
            "choices": [],
            "usage": usage.model_dump(),
        }
        yield f"data: {json.dumps(usage_payload, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Validate / extract the user question EAGERLY so a missing-user-message
    # error becomes a clean 400 rather than an abrupt mid-stream close.
    # Previously this lived inside `_stream_response`, but FastAPI's
    # StreamingResponse flushes headers (200 OK) before iterating the body —
    # an HTTPException raised mid-stream is not converted to a 4xx, the
    # connection just dies on the wire.
    question = _extract_user_question(req)

    if req.report:
        if req.stream:
            raise HTTPException(status_code=400, detail="report=true requires stream=false")
        agent = get_agent()
        result = agent.answer(
            question,
            temperature=req.temperature,
            top_p=req.top_p,
            max_new_tokens=req.max_tokens or 4096,
        )
        report_text = agent.generate_report(result)
        prompt_text = "\n".join(m.content for m in req.messages if getattr(m, "content", None))
        usage = _count_usage(agent, prompt_text, report_text or "")
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=_model_id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=report_text),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    if req.stream:
        return StreamingResponse(
            _stream_response(req, question),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming path: synchronous answer().
    agent = get_agent()
    result = agent.answer(
        question,
        temperature=req.temperature,
        top_p=req.top_p,
        max_new_tokens=req.max_tokens or 4096,
    )

    # Populate token usage. OpenAI semantics: prompt_tokens = the client's
    # input messages, completion_tokens = the returned answer. (The agent
    # internally consumes far more across its tool-calling rounds, but the
    # client-facing contract is input→output, which is what cost-tracking
    # SDK clients expect.) Counted with the model tokenizer when available;
    # falls back to a whitespace word count so usage is never silently zero.
    prompt_text = "\n".join(m.content for m in req.messages if getattr(m, "content", None))
    usage = _count_usage(agent, prompt_text, result.answer or "")

    # OpenAI finish_reason convention:
    #   "stop"   — model emitted a normal stop sequence
    #   "length" — generation hit a length limit. We map this to BOTH a
    #              timeout-cancellation (``result.cancelled``) and a
    #              ``max_round`` force-finish (``result.forced``), so SDK
    #              clients can distinguish a complete answer from a
    #              truncated one in either case.
    finish_reason = "length" if (result.cancelled or result.forced) else "stop"

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=_model_id,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result.answer),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


_STARTED_AT = time.time()


@app.get("/health")
def health() -> dict[str, Any]:
    from athena_r1 import __version__

    payload: dict[str, Any] = {
        "status": "ok",
        "model": _model_id,
        "initialized": _agent is not None,
        "version": __version__,
        "uptime_s": round(time.time() - _STARTED_AT, 1),
    }
    if _agent is not None:
        payload["agent"] = _agent.info()
    return payload


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "9000")),
    )


if __name__ == "__main__":
    main()
