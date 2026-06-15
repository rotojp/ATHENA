"""Pathological-event probes for the AG-UI translator.

Things the real engine *probably* won't do but easily *could* under bug
conditions — verify the translator degrades gracefully instead of crashing.

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


def _drive(async_gen):
    events = []

    async def consume():
        async for event in async_gen:
            events.append(event)

    asyncio.run(consume())
    return events


def _install(server, fn):
    class _SC:
        max_agent_level = 0

    class _SA:
        _core = _SC()
        answer_streaming = staticmethod(fn)

    server._agent = _SA()


# ────────────────────────────────────────────────────────────────────────


def test_tool_result_without_preceding_tool_call_does_not_crash():
    """Engine yields tool_result with no matching tool_call → translator
    silently drops it (open_tool_call_id .pop with default None)."""
    server = _import_translate()

    def orphan_result(question, **kw):
        yield _stub_event("round_start", round=1)
        yield _stub_event("tool_result", content="result with no parent")
        yield _stub_event("final_answer", content="done")

    _install(server, orphan_result)
    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    # No crash, RUN_FINISHED at the end.
    assert any(type(e).__name__ == "RunFinishedEvent" for e in events)


def test_tools_retrieved_without_tool_rag_query_does_not_crash():
    """Similar pairing: tools_retrieved with no matching tool_rag_query."""
    server = _import_translate()

    def orphan_retrieved(question, **kw):
        yield _stub_event("round_start", round=1)
        yield _stub_event("tools_retrieved", metadata={"tools": ["X", "Y"]})
        yield _stub_event("final_answer", content="ok")

    _install(server, orphan_retrieved)
    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    assert any(type(e).__name__ == "RunFinishedEvent" for e in events)


def test_tool_call_with_missing_tool_name_falls_back():
    """``tool_name`` missing from metadata → translator uses ``"tool"`` fallback,
    no crash."""
    server = _import_translate()

    def no_tool_name(question, **kw):
        yield _stub_event("round_start", round=1)
        # Note: metadata has no tool_name key
        yield _stub_event("tool_call", content='{"arguments":{}}', metadata={})
        yield _stub_event("final_answer", content="ok")

    _install(server, no_tool_name)
    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    tc_starts = [e for e in events if type(e).__name__ == "ToolCallStartEvent"]
    assert len(tc_starts) == 1, [type(e).__name__ for e in events]
    assert tc_starts[0].tool_call_name == "tool"


def test_tool_call_with_malformed_json_content_falls_back_to_empty_args():
    """``content`` is invalid JSON → args parsed as {}, no crash."""
    server = _import_translate()

    def bad_json(question, **kw):
        yield _stub_event("round_start", round=1)
        yield _stub_event(
            "tool_call",
            content="{not valid",
            metadata={"tool_name": "FDA_get_dose"},
        )
        yield _stub_event("final_answer", content="ok")

    _install(server, bad_json)
    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    assert any(type(e).__name__ == "ToolCallStartEvent" for e in events)
    assert any(type(e).__name__ == "RunFinishedEvent" for e in events)


def test_reasoning_with_empty_content_is_skipped():
    """Empty/whitespace reasoning shouldn't emit a TEXT_MESSAGE_START."""
    server = _import_translate()

    def whitespace_reasoning(question, **kw):
        yield _stub_event("round_start", round=1)
        yield _stub_event("reasoning", content="   \n\t  ")
        yield _stub_event("reasoning", content="")
        yield _stub_event("reasoning", content=None)
        yield _stub_event("final_answer", content="answer")

    _install(server, whitespace_reasoning)
    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    types_seen = [type(e).__name__ for e in events]
    # The 3 empty reasoning events should NOT produce TextMessage* events.
    # The final_answer produces its own TEXT_MESSAGE_*, so we expect exactly 1
    # TEXT_MESSAGE_START (the final answer's).
    n_text_start = sum(1 for t in types_seen if t == "TextMessageStartEvent")
    assert n_text_start == 1, (
        f"expected 1 TextMessageStart from final_answer, got {n_text_start}: {types_seen}"
    )


def test_force_finish_synthesized_answer_emits_text_message():
    """Production force-finish path: engine emits a normal reasoning round and
    THEN a `final_answer` carrying a synthesized answer the model never
    produced. The synthesized content is NOT inside the last reasoning, so the
    translator MUST emit a TextMessage for it — otherwise the UI's
    final-answer panel pulls from the last reasoning's speech segment and
    silently shows the incomplete trace instead of the recovered answer."""
    server = _import_translate()

    def force_finish(question, **kw):
        yield _stub_event("round_start", round=1)
        yield _stub_event("reasoning", content="<think>partial trace ...</think>")
        yield _stub_event(
            "final_answer",
            content="Synthesized answer based on partial reasoning.",
            metadata={"forced": True},
        )

    _install(server, force_finish)
    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    types_seen = [type(e).__name__ for e in events]
    # 2 TextMessage triplets: one for reasoning, one for the synthesized final.
    n_text_start = sum(1 for t in types_seen if t == "TextMessageStartEvent")
    assert n_text_start == 2, (
        f"expected 2 TextMessageStart (reasoning + synthesized final), got "
        f"{n_text_start}: {types_seen}"
    )
    # The synthesized text must be present in some Content event.
    contents = [
        getattr(e, "delta", "") for e in events if type(e).__name__ == "TextMessageContentEvent"
    ]
    assert any("Synthesized answer" in c for c in contents), contents
    # Critically, the synthesized answer must land in its OWN new round step
    # (round-2), NOT overwrite the model's partial trace in round-1.
    step_starts = [
        getattr(e, "step_name", "") for e in events if type(e).__name__ == "StepStartedEvent"
    ]
    assert any(s.endswith("/round-2") for s in step_starts), (
        f"expected a synthetic round-2 step for the recovery answer, got step_starts={step_starts}"
    )


def test_normal_final_answer_does_not_duplicate_reasoning():
    """Normal path: model emits "<think>...</think>[FinalAnswer]X" inline; engine
    streams the whole thing as `reasoning` and repeats "X" in `final_answer`.
    The translator must NOT re-emit "X" as a second TextMessage because the
    UI would then overwrite the round's reasoning (losing the <think>
    block) and double-render the answer."""
    server = _import_translate()

    def normal_flow(question, **kw):
        yield _stub_event("round_start", round=1)
        yield _stub_event(
            "reasoning",
            content="<think>reasoning here</think>The answer is amoxicillin 500mg TID for 7 days.",
        )
        yield _stub_event(
            "final_answer",
            content="The answer is amoxicillin 500mg TID for 7 days.",
        )

    _install(server, normal_flow)
    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    types_seen = [type(e).__name__ for e in events]
    n_text_start = sum(1 for t in types_seen if t == "TextMessageStartEvent")
    assert n_text_start == 1, (
        f"expected exactly 1 TextMessageStart (just the reasoning), got "
        f"{n_text_start}: {types_seen}"
    )


def test_unmatched_subagent_start_gets_closed_at_run_end():
    """Engine crash / progress_callback failure can leave a subagent_start
    event in the stream without a matching subagent_end. The translator
    must still emit a closing StepFinishedEvent at function exit so strict
    AG-UI clients observe a well-formed step-event tree.

    Regression: previously only ``open_round_step`` was tracked; orphaned
    ``/subagent`` steps would leave the client thinking the sub-agent is
    still executing forever.
    """
    server = _import_translate()

    def crash_before_end(question, **kw):
        # subagent_start fires, child events happen, but the engine then
        # raises before CallAgentProcessor can emit subagent_end.
        yield _stub_event("round_start", round=1, agent_id="main")
        yield _stub_event(
            "subagent_start",
            agent_id="main.sub-1",
            parent_agent_id="main",
            agent_level=1,
            content="task",
        )
        yield _stub_event("round_start", round=1, agent_id="main.sub-1", agent_level=1)
        yield _stub_event("reasoning", content="partial sub", agent_id="main.sub-1", agent_level=1)
        # Simulate engine error — no subagent_end, no final_answer.
        raise RuntimeError("simulated engine crash")

    _install(server, crash_before_end)
    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    types_seen = [type(e).__name__ for e in events]

    # The /subagent step must have its closing event.
    step_starts = [
        getattr(e, "step_name", "") for e in events if type(e).__name__ == "StepStartedEvent"
    ]
    step_finishes = [
        getattr(e, "step_name", "") for e in events if type(e).__name__ == "StepFinishedEvent"
    ]
    sub_starts = [s for s in step_starts if s.endswith("/subagent")]
    sub_finishes = [s for s in step_finishes if s.endswith("/subagent")]
    assert sub_starts and sub_finishes and sub_starts == sub_finishes, (
        f"unmatched /subagent step: starts={sub_starts}, finishes={sub_finishes}, types_seen={types_seen}"
    )
    # And the crash must still surface as a RunErrorEvent.
    assert any(t == "RunErrorEvent" for t in types_seen), types_seen


def test_subagent_force_finish_recovery_round_in_sub_scope():
    """Force-finish in a SUB-agent must produce its recovery round inside the
    sub's scope (step name starting with the sub's agent_id), not in main.

    Regression guard: agui_server's per-agent counters
    (last_round_num_per_agent) and the synthetic step name must be keyed by
    ``scope_label``, not by "main" — otherwise the sub's recovery answer
    lands in the main agent's rounds and the sub's final-text extraction
    picks up the model's incomplete trace instead.
    """
    server = _import_translate()

    def sub_force_finish(question, **kw):
        yield _stub_event("round_start", round=1, agent_id="main")
        yield _stub_event(
            "subagent_start",
            agent_id="main.sub-1",
            parent_agent_id="main",
            agent_level=1,
            content="task",
        )
        yield _stub_event("round_start", round=1, agent_id="main.sub-1", agent_level=1)
        yield _stub_event(
            "reasoning",
            content="<think>incomplete sub trace</think>",
            agent_id="main.sub-1",
            agent_level=1,
        )
        # Sub force-finish: content NOT in last reasoning → triggers recovery round.
        yield _stub_event(
            "final_answer",
            content="Sub synthesized recovery answer.",
            agent_id="main.sub-1",
            agent_level=1,
            metadata={"forced": True},
        )
        yield _stub_event(
            "subagent_end",
            agent_id="main.sub-1",
            parent_agent_id="main",
            agent_level=1,
            content="Sub synthesized recovery answer.",
        )
        # Main continues and finishes normally.
        yield _stub_event(
            "reasoning",
            content="<think>back in main</think>Final main answer.",
            agent_id="main",
            agent_level=0,
        )
        yield _stub_event(
            "final_answer",
            content="Final main answer.",
            agent_id="main",
            agent_level=0,
        )

    _install(server, sub_force_finish)
    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    step_starts = [
        getattr(e, "step_name", "") for e in events if type(e).__name__ == "StepStartedEvent"
    ]
    # The sub's recovery step MUST be in the sub's namespace.
    assert any(s.startswith("main.sub-1/") and s.endswith("/round-2") for s in step_starts), (
        f"sub recovery step missing or in wrong scope: step_starts={step_starts}"
    )
    # Main's rounds must still come through.
    assert any(s == "main/round-1" for s in step_starts), step_starts


def test_subagent_events_emit_step_pairs():
    """Sub-agent events should produce matched STEP_STARTED/FINISHED pairs."""
    server = _import_translate()

    def with_subagent(question, **kw):
        yield _stub_event("round_start", round=1, agent_id="main")
        yield _stub_event(
            "subagent_start",
            agent_id="main.sub-1",
            parent_agent_id="main",
            agent_level=1,
            content="child sub-question",
        )
        yield _stub_event("round_start", round=1, agent_id="main.sub-1", agent_level=1)
        yield _stub_event(
            "reasoning", content="child thinking", agent_id="main.sub-1", agent_level=1
        )
        yield _stub_event(
            "final_answer", content="child answer", agent_id="main.sub-1", agent_level=1
        )
        yield _stub_event(
            "subagent_end",
            agent_id="main.sub-1",
            parent_agent_id="main",
            content="child answer",
        )
        yield _stub_event("reasoning", content="back in main", agent_id="main")
        yield _stub_event("final_answer", content="final main", agent_id="main", agent_level=0)

    _install(server, with_subagent)
    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    types_seen = [type(e).__name__ for e in events]
    n_started = sum(1 for t in types_seen if t == "StepStartedEvent")
    n_finished = sum(1 for t in types_seen if t == "StepFinishedEvent")
    assert n_started == n_finished, (
        f"{n_started} STEP_STARTED vs {n_finished} STEP_FINISHED: {types_seen}"
    )
    # Should see at least 3 step pairs: main/round-1, sub-1/subagent, sub-1/round-1.
    assert n_started >= 3, types_seen


def test_messages_emitted_under_subagent_carry_subagent_id_in_msg_id():
    """The new ::-encoded message_id should preserve the sub-agent's agent_id."""
    server = _import_translate()

    def with_subagent(question, **kw):
        yield _stub_event("round_start", round=1, agent_id="main.sub-1", agent_level=1)
        yield _stub_event("reasoning", content="from child", agent_id="main.sub-1", agent_level=1)
        yield _stub_event("final_answer", content="ok", agent_id="main", agent_level=0)

    _install(server, with_subagent)
    events = _drive(server._translate_to_agui("Q?", "t", "r"))
    # Find the TEXT_MESSAGE_START emitted by the sub-agent's reasoning event.
    starts = [e for e in events if type(e).__name__ == "TextMessageStartEvent"]
    assert len(starts) >= 1
    # The sub-agent's reasoning came first, so the first TEXT_MESSAGE_START
    # should encode "main.sub-1".
    sub_msg_id = starts[0].message_id
    assert "main.sub-1" in sub_msg_id, f"expected agent_id in id, got {sub_msg_id!r}"


def test_pathological_agent_id_with_double_colon_separator_still_decodes_safely():
    """If agent_id itself contains '::' (defense-in-depth), decoding should
    still produce *some* string — not crash. The encoded id has 3+ parts."""
    sys.path.insert(0, "web")
    from agui_server import _msg_id, agent_id_from_event_id

    weird = "main::sub-1"  # someone passed an agent_id with the separator in it
    mid = _msg_id(weird)
    decoded = agent_id_from_event_id(mid)
    # We don't promise round-trip for this case — only that decoding doesn't crash.
    assert isinstance(decoded, str)
    assert decoded != ""


def test_engine_yields_event_with_agent_id_none_does_not_crash():
    """Defensive: the engine yields an event with agent_id=None."""
    server = _import_translate()

    def none_agent_id(question, **kw):
        ev = _stub_event("reasoning", content="weird")
        ev.agent_id = None  # type: ignore
        yield ev
        yield _stub_event("final_answer", content="ok")

    _install(server, none_agent_id)
    # Just verify no exception — translator may use None as a key, that's ok.
    try:
        events = _drive(server._translate_to_agui("Q?", "t", "r"))
    except Exception as e:  # noqa: BLE001
        # The translator should be defensive — fail the test loudly if it isn't.
        raise AssertionError(f"translator crashed on agent_id=None: {e!r}") from e
    # Should still terminate normally.
    types_seen = [type(e).__name__ for e in events]
    assert types_seen[-1] in ("RunFinishedEvent", "RunErrorEvent"), types_seen


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
