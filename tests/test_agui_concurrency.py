"""Concurrent-translator safety probe.

Two parallel ``_translate_to_agui`` calls against the same module-singleton
stub agent — verify each consumer sees only its own events. Catches:

  * State leak via module-level globals (``open_message_id`` etc.) — these
    are request-local but easy to accidentally refactor into globals.
  * Per-request ``cancel_event`` getting shared across runs.
  * Event delivery to wrong queue.

Run offline — no live vLLM / TU.
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
    ev = types.SimpleNamespace()
    ev.type = typ
    ev.content = kw.get("content")
    ev.round = kw.get("round")
    ev.agent_id = kw.get("agent_id", "main")
    ev.agent_level = kw.get("agent_level", 0)
    ev.parent_agent_id = kw.get("parent_agent_id")
    ev.metadata = kw.get("metadata", {})
    return ev


def test_two_concurrent_runs_dont_cross_leak_events():
    """Each consumer must only see events belonging to its own question.
    If the queues were shared, run-A's content would leak into run-B's stream."""
    server = _import_translate()

    # Shared stub: dispatches on the question string so we can prove
    # consumer A sees "A-content" only and consumer B sees "B-content" only.
    import time as _time

    def dispatch_by_question(question, **kw):
        tag = "A" if "AAAA" in question else "B"
        yield _stub_event("round_start", round=1)
        # Slow down a touch so the two runs are guaranteed to overlap.
        _time.sleep(0.02)
        yield _stub_event("reasoning", content=f"{tag}-reasoning-text")
        _time.sleep(0.02)
        yield _stub_event("final_answer", content=f"{tag}-final", agent_level=0)

    class _SC:
        max_agent_level = 0

    class _SA:
        _core = _SC()
        answer_streaming = staticmethod(dispatch_by_question)

    server._agent = _SA()

    async def consume(question, thread_id, run_id):
        events = []
        async for ev in server._translate_to_agui(question, thread_id, run_id):
            events.append(ev)
        return events

    async def drive_both():
        return await asyncio.gather(
            consume("AAAA-question", "t-A", "r-A"),
            consume("BBBB-question", "t-B", "r-B"),
        )

    a_events, b_events = asyncio.run(drive_both())

    a_text = "".join(
        getattr(e, "delta", "") for e in a_events if type(e).__name__ == "TextMessageContentEvent"
    )
    b_text = "".join(
        getattr(e, "delta", "") for e in b_events if type(e).__name__ == "TextMessageContentEvent"
    )

    # Each consumer should ONLY see its own tag, never the other's.
    assert "A-reasoning-text" in a_text or "A-final" in a_text, a_text
    assert "B-reasoning-text" not in a_text, f"B content leaked into A stream: {a_text!r}"
    assert "B-reasoning-text" in b_text or "B-final" in b_text, b_text
    assert "A-reasoning-text" not in b_text, f"A content leaked into B stream: {b_text!r}"

    # Both should report matching thread/run IDs in their RUN_STARTED.
    a_started = next(e for e in a_events if type(e).__name__ == "RunStartedEvent")
    b_started = next(e for e in b_events if type(e).__name__ == "RunStartedEvent")
    assert a_started.thread_id == "t-A" and a_started.run_id == "r-A"
    assert b_started.thread_id == "t-B" and b_started.run_id == "r-B"


def test_concurrent_message_ids_are_unique_per_run():
    """Two parallel runs must produce disjoint message_id sets — otherwise a
    frontend would render run-A's text into run-B's bubble."""
    server = _import_translate()

    def yields_one_msg(question, **kw):
        yield _stub_event("round_start", round=1)
        yield _stub_event("reasoning", content=f"text-for-{question[:10]}")
        yield _stub_event("final_answer", content="done", agent_level=0)

    class _SC:
        max_agent_level = 0

    class _SA:
        _core = _SC()
        answer_streaming = staticmethod(yields_one_msg)

    server._agent = _SA()

    async def consume(q, t, r):
        return [e async for e in server._translate_to_agui(q, t, r)]

    async def drive_both():
        return await asyncio.gather(
            consume("Q-one", "t1", "r1"),
            consume("Q-two", "t2", "r2"),
        )

    a, b = asyncio.run(drive_both())
    a_ids = {e.message_id for e in a if type(e).__name__ == "TextMessageStartEvent"}
    b_ids = {e.message_id for e in b if type(e).__name__ == "TextMessageStartEvent"}
    assert a_ids and b_ids
    overlap = a_ids & b_ids
    assert not overlap, f"message_id collision across concurrent runs: {overlap}"


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
