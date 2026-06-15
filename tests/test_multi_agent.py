"""Multi-agent streaming tests.

Verifies:
    1. RoundEvent carries agent_id / agent_level / parent_agent_id
    2. When max_agent_level=0 (default), no subagent_start events fire
    3. When max_agent_level>=1, the wiring is in place — we don't force the
       model to call CallAgent (it may or may not), but we DO verify the
       schema/plumbing is correct via a direct unit-test path.

Note: Whether the model actually emits `<tool_call>CallAgent</tool_call>` is
non-deterministic and depends on training. The pytest fixtures default to
the live cluster servers; if you want to force a multi-agent path
deterministically, run the unit test below which directly invokes
``CallAgentProcessor.process()``.
"""

import sys


def test_default_no_subagent_events(agent):
    """Main agent never emits subagent_start unless max_agent_level > 0."""
    counts: dict[str, int] = {}
    seen_agent_ids: set[str] = set()
    for ev in agent.answer_streaming(
        "What FDA-labeled contraindications exist for metformin?",
        temperature=0.7,
        max_round=6,
    ):
        counts[ev.type] = counts.get(ev.type, 0) + 1
        seen_agent_ids.add(ev.agent_id)
    # The default agent runs entirely as "main" with no subagents.
    assert seen_agent_ids == {"main"}, seen_agent_ids
    assert "subagent_start" not in counts, counts


def test_event_schema_fields_present(agent):
    """Every emitted event has the new schema fields populated."""
    sample = None
    for ev in agent.answer_streaming(
        "What FDA-labeled contraindications exist for metformin?",
        temperature=0.7,
        max_round=6,
    ):
        if sample is None:
            sample = ev
            break
    assert sample is not None
    assert sample.agent_id == "main"
    assert sample.agent_level == 0
    assert sample.parent_agent_id is None


def test_callagent_processor_emits_subagent_events_unit(monkeypatch, agent):
    """Direct unit-level: bypass the model and exercise the recursion path.

    Patches CallAgentProcessor.process to not actually recurse (which would
    take 30+ s of vLLM time), but verify the start/end events would fire
    and the agent_id naming convention works.
    """
    from athena_r1.tool_processors import CallAgentProcessor

    proc = CallAgentProcessor(agent._core)
    agent._core.max_agent_level = 2

    captured: list[dict] = []
    context = {
        "call_agent": True,
        "call_agent_level": 0,
        "agent_id": "main",
        "progress_callback": lambda ev: captured.append(ev),
        "max_new_tokens": 1024,
    }

    # Stub out the recursive call so the test is fast.
    def fake_run(message, **kwargs):
        return "stub child final answer", []

    monkeypatch.setattr(agent._core, "run_multistep_agent", fake_run)

    result, ctx_updates = proc.process(
        {"name": "CallAgent", "arguments": {"solution": "do thing X"}},
        context,
    )

    assert result == "stub child final answer"
    types = [e["type"] for e in captured]
    assert "subagent_start" in types and "subagent_end" in types
    start = next(e for e in captured if e["type"] == "subagent_start")
    end = next(e for e in captured if e["type"] == "subagent_end")
    assert start["agent_id"] == "main.sub-1"
    assert start["parent_agent_id"] == "main"
    assert start["agent_level"] == 1
    assert end["content"] == "stub child final answer"
    assert ctx_updates["special_tool_call"] == "CallAgent"


def test_callagent_disabled_returns_error(agent):
    """When call_agent=False or max_agent_level=0, the processor short-circuits."""
    from athena_r1.tool_processors import CallAgentProcessor

    agent._core.max_agent_level = 0
    proc = CallAgentProcessor(agent._core)
    captured: list[dict] = []
    context = {
        "call_agent": False,
        "call_agent_level": 0,
        "agent_id": "main",
        "progress_callback": lambda ev: captured.append(ev),
    }
    result, _ = proc.process(
        {"name": "CallAgent", "arguments": {"solution": "do thing X"}},
        context,
    )
    assert "disabled" in result.lower()
    assert captured == []  # no subagent events when disabled


def test_subagent_final_answer_does_not_end_run(monkeypatch, agent):
    """Sub-agent ``final_answer`` events MUST NOT short-circuit the streaming
    consumer. The engine emits one ``final_answer`` per recursive
    ``run_multistep_agent`` call (each sub-agent emits its own); only the
    main agent's final_answer (``agent_level == 0``) signals end-of-run.

    Regression: an earlier version returned on the first ``final_answer``
    seen, which silently dropped ``subagent_end`` + the rest of the main
    agent's rounds + the actual top-level final_answer for every multi-
    agent run.
    """
    sequence = [
        {"type": "round_start", "round": 1, "agent_id": "main", "agent_level": 0},
        {
            "type": "subagent_start",
            "agent_id": "main.sub-1",
            "parent_agent_id": "main",
            "agent_level": 1,
            "content": "child task",
        },
        {"type": "round_start", "round": 1, "agent_id": "main.sub-1", "agent_level": 1},
        {
            "type": "reasoning",
            "content": "child thinking",
            "agent_id": "main.sub-1",
            "agent_level": 1,
        },
        {
            "type": "final_answer",
            "content": "child answer",
            "agent_id": "main.sub-1",
            "agent_level": 1,
        },
        {
            "type": "subagent_end",
            "agent_id": "main.sub-1",
            "agent_level": 1,
            "parent_agent_id": "main",
            "content": "child answer",
        },
        {"type": "reasoning", "content": "back in main", "agent_id": "main", "agent_level": 0},
        {
            "type": "final_answer",
            "content": "top-level answer",
            "agent_id": "main",
            "agent_level": 0,
        },
    ]

    def fake_run(message, **kwargs):
        cb = kwargs.get("progress_callback")
        if cb is not None:
            for ev in sequence:
                cb(dict(ev))
        return "top-level answer", []

    monkeypatch.setattr(agent._core, "run_multistep_agent", fake_run)

    received_types = []
    for ev in agent.answer_streaming("test question please"):
        received_types.append((ev.type, ev.agent_id, ev.agent_level))

    # We must see EVERY event from the sequence — especially the events that
    # come AFTER the sub-agent's final_answer (subagent_end, parent reasoning,
    # top-level final_answer).
    assert ("subagent_end", "main.sub-1", 1) in received_types, received_types
    assert ("reasoning", "main", 0) in received_types, received_types
    assert ("final_answer", "main", 0) in received_types, received_types
    # And the sub-agent's final_answer is still yielded (just doesn't terminate
    # the iteration).
    assert ("final_answer", "main.sub-1", 1) in received_types, received_types


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
