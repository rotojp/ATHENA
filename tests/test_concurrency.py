"""Concurrent-call safety: two simultaneous answer() calls on the same agent
should not cross-cancel via the timeout mechanism.
"""

from __future__ import annotations

import sys
import threading
import time


def test_concurrent_timeouts_dont_interfere(monkeypatch, agent):
    """Call A times out, Call B doesn't — verify B is not cancelled."""

    # Stub the engine to honor each call's own cancel_event.
    observed_cancels: list[bool] = []

    def stub_runner(message, **kwargs):
        cancel_event = kwargs.pop("cancel_event", None)
        # Different sleep durations: short and long
        target = 1.0 if "short" in message else 5.0
        elapsed = 0.0
        while elapsed < target:
            time.sleep(0.1)
            elapsed += 0.1
            if cancel_event is not None and cancel_event.is_set():
                observed_cancels.append(True)
                return f"cancelled at {elapsed:.1f}s", []
        observed_cancels.append(False)
        return f"finished {message}", []

    monkeypatch.setattr(agent._core, "run_multistep_agent", stub_runner)

    results: dict[str, object] = {}

    def call_with_timeout(label: str, q: str, timeout: float) -> None:
        try:
            results[label] = agent.answer(q, timeout=timeout).answer
        except Exception as e:  # noqa: BLE001
            results[label] = f"ERR {e!r}"

    # Call A: short question with tight timeout → should time out
    # Call B: long question with generous timeout → should complete
    ta = threading.Thread(
        target=call_with_timeout,
        args=("A", "short question please answer me carefully", 0.3),
    )
    tb = threading.Thread(
        target=call_with_timeout,
        args=("B", "long question please answer me carefully", 30.0),
    )
    ta.start()
    tb.start()
    ta.join(timeout=15.0)
    tb.join(timeout=15.0)

    # Both threads should have finished
    assert "A" in results and "B" in results, results
    # A was cancelled (timeout 0.3s), B was NOT cancelled (had 30s vs 5s work)
    assert "cancelled" in results["A"], f"A should have been cancelled: {results['A']!r}"
    assert "finished" in results["B"], f"B should have finished cleanly: {results['B']!r}"


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
