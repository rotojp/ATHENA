"""Unit tests for two new features: agent.answer(timeout=...) and OpenAI
multi-turn prompt assembly.

These tests don't require a live vLLM — they unit-test the timeout plumbing
by stubbing run_multistep_agent, and the multi-turn helper directly.
"""

from __future__ import annotations

import sys
import time


def test_answer_timeout_calls_cancel(monkeypatch, agent):
    """When timeout fires, the per-call cancel_event should be set."""

    captured = {}

    # Make run_multistep_agent sleep way past the timeout.
    def slow_runner(message, **kwargs):
        # Pop the per-call cancel event the wrapper passes in.
        cancel_event = kwargs.pop("cancel_event", None)
        captured["event"] = cancel_event
        # Honor cancel cooperatively — like the real engine does.
        for _ in range(60):
            time.sleep(0.5)
            if cancel_event is not None and cancel_event.is_set():
                return "cancelled-partial-answer", []
        return "should-not-reach", []

    monkeypatch.setattr(agent._core, "run_multistep_agent", slow_runner)

    t0 = time.monotonic()
    result = agent.answer("This is a real question about something", timeout=2.0)
    elapsed = time.monotonic() - t0

    # Should return within ~2s + grace (well under the 30s the stub would take).
    assert elapsed < 10.0, f"timeout ignored: took {elapsed:.1f}s"
    # The wrapper should have supplied a cancel_event and set it.
    assert captured["event"] is not None, "wrapper did not pass cancel_event"
    assert captured["event"].is_set() is True
    # The cooperative response should bubble back as the answer.
    assert "cancelled" in result.answer


def test_answer_no_timeout_runs_normally(monkeypatch, agent):
    """timeout=None (default) shouldn't activate the timeout wrapper at all."""
    calls = {"n": 0}

    def quick_runner(message, **kwargs):
        calls["n"] += 1
        return "fast answer", []

    monkeypatch.setattr(agent._core, "run_multistep_agent", quick_runner)
    result = agent.answer("This is a real question about something")
    assert calls["n"] == 1
    assert result.answer == "fast answer"
    assert result.cancelled is False


def test_answer_cancelled_flag_on_timeout(monkeypatch, agent):
    """When timeout fires, AnswerResult.cancelled must be True."""

    def slow_runner(message, **kwargs):
        cancel_event = kwargs.pop("cancel_event", None)
        for _ in range(60):
            time.sleep(0.5)
            if cancel_event is not None and cancel_event.is_set():
                return "partial", []
        return "should-not-reach", []

    monkeypatch.setattr(agent._core, "run_multistep_agent", slow_runner)
    result = agent.answer("This is a real question about something", timeout=1.5)
    assert result.cancelled is True
    assert "partial" in result.answer


def test_answer_no_cancel_when_fast(monkeypatch, agent):
    """A fast call inside the timeout budget should have cancelled=False."""

    def quick_runner(message, **kwargs):
        return "done in 10ms", []

    monkeypatch.setattr(agent._core, "run_multistep_agent", quick_runner)
    result = agent.answer("This is a real question about something", timeout=60.0)
    assert result.cancelled is False
    assert result.answer == "done in 10ms"


def test_openai_server_multiturn_history_folded():
    """_extract_user_question() should fold prior turns into the prompt."""
    sys.path.insert(0, "web")
    from openai_server import ChatCompletionRequest, ChatMessage, _extract_user_question

    req = ChatCompletionRequest(
        model="athena-r1",
        messages=[
            ChatMessage(role="user", content="What is metformin used for?"),
            ChatMessage(role="assistant", content="Metformin treats type 2 diabetes."),
            ChatMessage(role="user", content="What's the renal dose adjustment?"),
        ],
    )
    prompt = _extract_user_question(req)
    # Folded prompt should contain prior turns + the new question
    assert "Conversation so far:" in prompt
    assert "user: What is metformin used for?" in prompt
    assert "assistant: Metformin treats type 2 diabetes." in prompt
    assert "New user question: What's the renal dose adjustment?" in prompt


def test_openai_server_singleturn_skips_history():
    """Single-message requests should NOT prepend the history scaffold."""
    sys.path.insert(0, "web")
    from openai_server import ChatCompletionRequest, ChatMessage, _extract_user_question

    req = ChatCompletionRequest(
        model="athena-r1",
        messages=[ChatMessage(role="user", content="Single message only.")],
    )
    prompt = _extract_user_question(req)
    assert prompt == "Single message only."


def test_openai_server_system_message_preserved():
    """System messages should be tagged as [system instruction] in history."""
    sys.path.insert(0, "web")
    from openai_server import ChatCompletionRequest, ChatMessage, _extract_user_question

    req = ChatCompletionRequest(
        model="athena-r1",
        messages=[
            ChatMessage(role="system", content="Be a careful clinician."),
            ChatMessage(role="user", content="First Q"),
            ChatMessage(role="assistant", content="First A"),
            ChatMessage(role="user", content="Follow-up Q"),
        ],
    )
    prompt = _extract_user_question(req)
    assert "[system instruction] Be a careful clinician." in prompt
    assert "New user question: Follow-up Q" in prompt


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
