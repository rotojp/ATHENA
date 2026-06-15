"""OpenAI-compat server error path tests.

After refactoring ``_extract_user_question`` to delegate to the shared
``web/_messages.fold_messages_to_prompt`` helper, verify the HTTP 400
contract still holds for all the cases the old code rejected:

  * empty messages array
  * only assistant / system messages (no user)
  * whitespace-only user content (new — was silently passed through before)

Plus a happy-path single-turn through ``_extract_user_question``.

Run offline — no FastAPI server needed, just the function.
"""

from __future__ import annotations

import sys

import pytest


def _import():
    sys.path.insert(0, "src")
    sys.path.insert(0, "web")
    from fastapi import HTTPException
    from openai_server import (
        ChatCompletionRequest,
        ChatMessage,
        _extract_user_question,
    )

    return ChatCompletionRequest, ChatMessage, _extract_user_question, HTTPException


def test_empty_messages_raises_400():
    Req, Msg, extract, HTTPException = _import()
    with pytest.raises(HTTPException) as exc:
        extract(Req(messages=[]))
    assert exc.value.status_code == 400
    assert "no user message" in exc.value.detail


def test_only_assistant_messages_raises_400():
    Req, Msg, extract, HTTPException = _import()
    req = Req(messages=[Msg(role="assistant", content="Hi, ask me anything")])
    with pytest.raises(HTTPException) as exc:
        extract(req)
    assert exc.value.status_code == 400


def test_only_system_message_raises_400():
    Req, Msg, extract, HTTPException = _import()
    req = Req(messages=[Msg(role="system", content="Be helpful")])
    with pytest.raises(HTTPException) as exc:
        extract(req)
    assert exc.value.status_code == 400


def test_whitespace_only_user_content_raises_400():
    """Regression for the recent fold-helper fix: whitespace-only no longer
    silently passes through to the model — it errors fast."""
    Req, Msg, extract, HTTPException = _import()
    req = Req(messages=[Msg(role="user", content="   \t\n  ")])
    with pytest.raises(HTTPException) as exc:
        extract(req)
    assert exc.value.status_code == 400


def test_single_user_message_returns_content_verbatim():
    Req, Msg, extract, _ = _import()
    req = Req(messages=[Msg(role="user", content="What is metformin used for?")])
    out = extract(req)
    assert out == "What is metformin used for?"
    # Single-turn should NOT prepend the multi-turn scaffold.
    assert "Conversation so far" not in out


def test_multi_turn_folds_history_into_prompt():
    Req, Msg, extract, _ = _import()
    req = Req(
        messages=[
            Msg(role="user", content="First question"),
            Msg(role="assistant", content="First answer"),
            Msg(role="user", content="Follow-up about that drug"),
        ]
    )
    out = extract(req)
    assert "Conversation so far:" in out
    assert "user: First question" in out
    assert "assistant: First answer" in out
    assert "New user question: Follow-up about that drug" in out


def test_request_validation_rejects_bad_sampling_params():
    """Pydantic field validators reject negative max_tokens and out-of-range
    temperature/top_p at the request boundary so vLLM doesn't error opaquely
    on garbage from the wire."""
    from pydantic import ValidationError

    Req, Msg, _, _ = _import()
    msgs = [Msg(role="user", content="hi there")]

    # Negative max_tokens.
    with pytest.raises(ValidationError):
        Req(messages=msgs, max_tokens=-10)
    # Zero max_tokens.
    with pytest.raises(ValidationError):
        Req(messages=msgs, max_tokens=0)
    # temperature out of [0, 2].
    with pytest.raises(ValidationError):
        Req(messages=msgs, temperature=-0.1)
    with pytest.raises(ValidationError):
        Req(messages=msgs, temperature=2.5)
    # top_p outside (0, 1].
    with pytest.raises(ValidationError):
        Req(messages=msgs, top_p=0.0)
    with pytest.raises(ValidationError):
        Req(messages=msgs, top_p=1.5)
    # Valid edge cases still pass.
    Req(messages=msgs, max_tokens=None)
    Req(messages=msgs, max_tokens=1)
    Req(messages=msgs, temperature=0.0)
    Req(messages=msgs, temperature=2.0)
    Req(messages=msgs, top_p=1.0)


def test_stream_opening_chunk_sets_role_assistant():
    """OpenAI streaming spec: the FIRST delta chunk should include
    ``role: "assistant"`` so SDK clients can identify the message. Strict
    consumers (langchain.OpenAI, older openai-python builds) drop subsequent
    content if role is never set on the first delta."""
    import json as _json
    import sys as _sys
    import types as _types

    _sys.path.insert(0, "src")
    _sys.path.insert(0, "web")
    import openai_server

    # Stub out get_agent + answer_streaming so we don't need vLLM live.
    def _fake_answer_streaming(question, **kw):
        # Minimal: a single round_start then a tiny final_answer so the
        # generator terminates promptly.
        ev = _types.SimpleNamespace
        yield ev(
            type="round_start",
            round=1,
            content=None,
            agent_id="main",
            agent_level=0,
            parent_agent_id=None,
            metadata={},
        )
        yield ev(
            type="final_answer",
            content="ok",
            agent_id="main",
            agent_level=0,
            parent_agent_id=None,
            metadata={},
        )

    class _FakeAgent:
        _initialized = True
        answer_streaming = staticmethod(_fake_answer_streaming)

    openai_server._agent = _FakeAgent()

    Req, Msg, _, _ = _import()
    req = Req(messages=[Msg(role="user", content="hi there")], stream=True)
    # _stream_response now takes the pre-extracted question explicitly.
    chunks = list(openai_server._stream_response(req, "hi there"))
    # First non-empty data: line is the opening chunk.
    first_data = next(c for c in chunks if c.startswith("data: ") and "[DONE]" not in c)
    payload = _json.loads(first_data[len("data: ") :].strip())
    delta = payload["choices"][0]["delta"]
    assert delta.get("role") == "assistant", (
        f"opening chunk must set role=assistant, got delta={delta}"
    )


def test_stream_missing_user_message_returns_4xx_not_broken_stream():
    """Regression: ``_extract_user_question`` used to be called INSIDE the
    body generator, so its HTTPException raised AFTER StreamingResponse
    flushed its 200 OK headers — the client saw an abrupt connection close
    instead of a 4xx with the actual error.

    The endpoint now extracts the question eagerly before deciding to
    stream, so a bad request becomes a clean HTTPException at the
    chat_completions level. We exercise that path through FastAPI's
    TestClient.
    """
    import sys as _sys

    _sys.path.insert(0, "src")
    _sys.path.insert(0, "web")
    import openai_server
    from fastapi.testclient import TestClient

    # Stub _agent so get_agent() doesn't try to load a model.
    class _FakeAgent:
        _initialized = True

    openai_server._agent = _FakeAgent()

    client = TestClient(openai_server.app)

    # Streaming request with ONLY an assistant message (no user). Old code
    # would return 200 with an empty/aborted SSE; new code returns 400.
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "athena-r1",
            "messages": [{"role": "assistant", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 400, (
        f"expected 400 for missing user message, got {resp.status_code}: {resp.text[:300]}"
    )


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
