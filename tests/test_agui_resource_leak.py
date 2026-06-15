"""Resource-leak smoke test for the AG-UI server's translator.

Drives ``_translate_to_agui`` N=20 times with a fast stub engine and checks:

  * the active thread count does not grow unboundedly across runs
    (worker threads exit at SENTINEL, queues get GC'd)
  * memory (RSS) growth between run #1 and run #N is below a modest budget
    (~30 MB) — catches accidental retention of conversation/queue state
    on a per-request basis

This test runs offline (no vLLM / TU needed).
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import threading
import time
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
    ev.parent_agent_id = None
    ev.metadata = kw.get("metadata", {})
    return ev


def _install_short_run_stub(server_mod):
    """Stub agent whose answer_streaming yields 3 events then returns."""

    def short_run(question, *, external_cancel_event=None, **kw):
        yield _stub_event("round_start", round=1)
        yield _stub_event("reasoning", content=f"Q={question[:30]}")
        yield _stub_event("final_answer", content="done", agent_level=0)

    class _SC:
        max_agent_level = 0

    class _SA:
        _core = _SC()
        answer_streaming = staticmethod(short_run)

    server_mod._agent = _SA()


def _drain(async_gen):
    events = []

    async def consume():
        async for event in async_gen:
            events.append(event)

    asyncio.run(consume())
    return events


def _rss_mb() -> float:
    """Self-RSS in MB. Linux /proc/self/status VmRSS."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def test_thread_count_bounded_across_runs():
    """Active threads should not grow with N. Each run's worker thread must
    terminate after SENTINEL — otherwise we leak threads per request."""
    server = _import_translate()
    _install_short_run_stub(server)

    # Warm-up: run once so any one-time-only threads (executor pool init) appear.
    _drain(server._translate_to_agui("warmup", "t0", "r0"))
    time.sleep(0.5)  # let runner threads exit
    gc.collect()
    threads_before = threading.active_count()

    N = 20
    for i in range(N):
        events = _drain(server._translate_to_agui(f"Q{i}", f"t{i}", f"r{i}"))
        # Sanity: each run produced a complete event sequence.
        types_seen = {type(e).__name__ for e in events}
        assert "RunStartedEvent" in types_seen
        assert "RunFinishedEvent" in types_seen

    # Allow worker threads to settle.
    time.sleep(1.0)
    gc.collect()
    threads_after = threading.active_count()

    # The executor's ThreadPoolExecutor may reuse threads, so 0 growth is
    # ideal but a small constant overhead is OK. Reject >5 thread growth as
    # a leak signal (would imply each request strands a thread).
    growth = threads_after - threads_before
    assert growth <= 5, (
        f"thread count grew by {growth} across {N} runs "
        f"({threads_before} → {threads_after}) — possible leak"
    )


def test_memory_growth_bounded_across_runs():
    """RSS should not balloon across N requests. Catches per-request retention
    of conversation buffers, queue items, etc."""
    if not os.path.exists("/proc/self/status"):
        # Non-Linux — skip; the test is Linux-specific.
        import pytest

        pytest.skip("requires /proc/self/status (Linux)")

    server = _import_translate()
    _install_short_run_stub(server)

    # Warm-up to amortize first-run import / lazy-init costs.
    for _ in range(3):
        _drain(server._translate_to_agui("warmup", "tw", "rw"))
    gc.collect()
    rss_before = _rss_mb()

    N = 20
    for i in range(N):
        _drain(server._translate_to_agui(f"Question number {i}", f"t{i}", f"r{i}"))

    gc.collect()
    rss_after = _rss_mb()
    growth_mb = rss_after - rss_before
    # 30 MB budget — generous, catches genuine leaks.
    assert growth_mb < 30, (
        f"RSS grew {growth_mb:.1f} MB over {N} runs "
        f"({rss_before:.1f} → {rss_after:.1f} MB) — possible leak"
    )


def test_no_orphan_state_after_error_paths():
    """If the engine raises, the translator must still emit RUN_ERROR and
    not leave open_message_id / open_tool_call_id state dangling in the
    module. We run an error path then a happy path and confirm the second
    run's events are well-formed (would not be if state leaked between
    runs — open_message_id is request-local but verify defensively)."""
    server = _import_translate()

    def boom(question, *, external_cancel_event=None, **kw):
        yield _stub_event("round_start", round=1)
        raise ConnectionError("simulated upstream death")

    def healthy(question, *, external_cancel_event=None, **kw):
        yield _stub_event("round_start", round=1)
        yield _stub_event("reasoning", content="thinking")
        yield _stub_event("final_answer", content="answer", agent_level=0)

    class _SC:
        max_agent_level = 0

    class _SA:
        _core = _SC()
        answer_streaming = staticmethod(boom)

    server._agent = _SA()
    bad_events = _drain(server._translate_to_agui("bad?", "t-bad", "r-bad"))
    bad_types = [type(e).__name__ for e in bad_events]
    assert "RunErrorEvent" in bad_types, bad_types

    # Swap to healthy stub. Per-request state shouldn't carry over.
    _SA.answer_streaming = staticmethod(healthy)
    server._agent = _SA()
    good_events = _drain(server._translate_to_agui("good?", "t-good", "r-good"))
    good_types = [type(e).__name__ for e in good_events]
    # Verify the good run finished cleanly.
    assert good_types[-1] == "RunFinishedEvent", good_types
    # And the STEP_STARTED count matches STEP_FINISHED.
    n_started = sum(1 for t in good_types if t == "StepStartedEvent")
    n_finished = sum(1 for t in good_types if t == "StepFinishedEvent")
    assert n_started == n_finished, f"step mismatch on healthy run after error: {good_types}"


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
