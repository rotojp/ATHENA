"""Upstream-failure handling for the AG-UI server.

These tests stub out the engine's ``answer_streaming`` so we can simulate the
ways vLLM / ToolUniverse can go wrong — connection refused, 5xx, hanging
forever, raising mid-stream — and verify the server emits a clean
``RUN_ERROR`` instead of leaking compute or hanging the connection.

No live vLLM / TU required.
"""

from __future__ import annotations

import asyncio
import sys
import types


def _import_translate():
    sys.path.insert(0, "src")
    sys.path.insert(0, "web")
    import agui_server

    return agui_server


def _stub_event(typ, **kw):
    """Return a duck-typed RoundEvent for the translator."""
    ev = types.SimpleNamespace()
    ev.type = typ
    ev.content = kw.get("content")
    ev.round = kw.get("round")
    ev.agent_id = kw.get("agent_id", "main")
    ev.agent_level = kw.get("agent_level", 0)
    ev.parent_agent_id = kw.get("parent_agent_id")
    ev.metadata = kw.get("metadata", {})
    return ev


def _drive(async_gen):
    """Drain an async generator to a list of events, synchronously."""
    events = []

    async def consume():
        async for event in async_gen:
            events.append(event)

    asyncio.run(consume())
    return events


def _install_stub_agent(server_mod, answer_streaming_fn):
    """Make get_agent() return a stub whose answer_streaming = fn."""

    class _StubCore:
        max_agent_level = 0

    class _StubAgent:
        def __init__(self):
            self._core = _StubCore()
            self.answer_streaming = answer_streaming_fn

    server_mod._agent = _StubAgent()
    return server_mod._agent


def test_engine_raises_connection_error_emits_run_error():
    """vLLM unreachable → answer_streaming raises ConnectionError → RUN_ERROR."""
    server = _import_translate()

    def boom(question, **kw):
        raise ConnectionError("vLLM at http://localhost:9999 refused connection")
        yield  # never reached, but makes this a generator

    _install_stub_agent(server, boom)

    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    types_seen = [type(e).__name__ for e in events]
    assert "RunStartedEvent" in types_seen
    assert "RunErrorEvent" in types_seen, f"no RUN_ERROR, got: {types_seen}"
    # The error message should surface the underlying failure.
    err = next(e for e in events if type(e).__name__ == "RunErrorEvent")
    assert "refused" in err.message.lower() or "vllm" in err.message.lower(), err.message


def test_engine_raises_after_partial_stream():
    """Engine yields a few events then dies → still RUN_ERROR, not a hang."""
    server = _import_translate()

    def partial(question, **kw):
        yield _stub_event("round_start", round=1)
        yield _stub_event("reasoning", content="thinking...")
        raise TimeoutError("tooluniverse request timed out after 60s")

    _install_stub_agent(server, partial)

    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    types_seen = [type(e).__name__ for e in events]
    # We should still see the early events the engine did emit.
    assert "StepStartedEvent" in types_seen, types_seen
    assert "TextMessageStartEvent" in types_seen, types_seen
    # ...and a clean error termination.
    assert "RunErrorEvent" in types_seen, types_seen
    err = next(e for e in events if type(e).__name__ == "RunErrorEvent")
    assert "timed out" in err.message.lower(), err.message


def test_engine_returns_no_events_emits_run_finished():
    """Engine completes with zero events (unusual but legal). RUN_FINISHED, no crash."""
    server = _import_translate()

    def empty(question, **kw):
        return
        yield  # makes this a generator

    _install_stub_agent(server, empty)

    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    types_seen = [type(e).__name__ for e in events]
    assert types_seen[0] == "RunStartedEvent"
    assert types_seen[-1] == "RunFinishedEvent", types_seen


def test_orphan_round_step_gets_finished_on_error():
    """If engine errors mid-round, the open STEP_STARTED still gets a STEP_FINISHED
    (otherwise frontends mis-pair their step trees)."""
    server = _import_translate()

    def opens_step_then_dies(question, **kw):
        yield _stub_event("round_start", round=1)  # opens step
        raise RuntimeError("upstream died with open step")

    _install_stub_agent(server, opens_step_then_dies)

    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    types_seen = [type(e).__name__ for e in events]
    n_started = sum(1 for t in types_seen if t == "StepStartedEvent")
    n_finished = sum(1 for t in types_seen if t == "StepFinishedEvent")
    assert n_started == n_finished, (
        f"{n_started} STEP_STARTED vs {n_finished} STEP_FINISHED: {types_seen}"
    )
    assert "RunErrorEvent" in types_seen, types_seen


def test_engine_yields_only_final_answer_emits_run_finished_cleanly():
    """Minimal happy path: just a final_answer event → assistant message + RUN_FINISHED."""
    server = _import_translate()

    def final_only(question, **kw):
        yield _stub_event("final_answer", content="The answer is X.", agent_level=0)

    _install_stub_agent(server, final_only)

    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    types_seen = [type(e).__name__ for e in events]
    assert "TextMessageStartEvent" in types_seen
    assert "TextMessageContentEvent" in types_seen
    assert "TextMessageEndEvent" in types_seen
    assert types_seen[-1] == "RunFinishedEvent"


# Note: an additional "engine hangs, client aborts" test lives in
# tests/test_agui_server.py::test_client_abort_does_not_hang_server because it
# requires a real asyncio event loop (FastAPI's) for the executor to pump the
# runner thread — pytest-driven asyncio.run() returns before the executor
# even starts. The live integration test there covers that path.


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
