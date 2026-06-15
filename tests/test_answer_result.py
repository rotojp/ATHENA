"""Tests for the AnswerResult dataclass + helpers."""

from __future__ import annotations

import os
import sys


def test_extract_tools_used_basic():
    os.environ["TOOLUNIVERSE_API"] = "http://0.0.0.0:8080"
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1

    conv = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": ('preamble <tool_call>{"name": "FDA_dose", "arguments": {}}</tool_call>'),
        },
        {"role": "tool", "content": "result"},
        {
            "role": "assistant",
            "content": ('<tool_call>{"name": "ChEMBL_act", "arguments": {"x": 1}}</tool_call>'),
        },
    ]
    tools = AthenaR1._extract_tools_used(conv)
    assert tools == ["FDA_dose", "ChEMBL_act"]


def test_extract_tools_used_filters_meta_tools():
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1

    conv = [
        {
            "role": "assistant",
            "content": (
                '<tool_call>{"name": "Tool_RAG", "arguments": {}}</tool_call>'
                '<tool_call>{"name": "CallAgent", "arguments": {}}</tool_call>'
                '<tool_call>{"name": "Finish", "arguments": {}}</tool_call>'
                '<tool_call>{"name": "OpenTargets_query", "arguments": {}}</tool_call>'
            ),
        },
    ]
    tools = AthenaR1._extract_tools_used(conv)
    assert tools == ["OpenTargets_query"]


def test_extract_tools_used_dedupes_across_turns():
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1

    conv = [
        {
            "role": "assistant",
            "content": '<tool_call>{"name": "FDA_dose", "arguments": {}}</tool_call>',
        },
        {"role": "tool", "content": "r1"},
        {
            "role": "assistant",
            "content": '<tool_call>{"name": "FDA_dose", "arguments": {"limit": 10}}</tool_call>',
        },
    ]
    tools = AthenaR1._extract_tools_used(conv)
    assert tools == ["FDA_dose"]


def test_extract_tools_used_handles_malformed_json():
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1

    conv = [
        {"role": "assistant", "content": "<tool_call>{not valid json}</tool_call>"},
        {
            "role": "assistant",
            "content": '<tool_call>{"name": "GoodTool", "arguments": {}}</tool_call>',
        },
    ]
    tools = AthenaR1._extract_tools_used(conv)
    assert tools == ["GoodTool"]


def test_answer_result_default_fields():
    sys.path.insert(0, "src")
    from athena_r1 import AnswerResult

    r = AnswerResult(answer="x")
    assert r.answer == "x"
    assert r.conversation == []
    assert r.rounds_used == 0
    assert r.elapsed_s == 0.0
    assert r.tools_used == []
    assert r.cancelled is False
    # ``forced`` defaults False — only True when ``max_round`` was exhausted.
    assert r.forced is False


def test_answer_result_forced_detection(monkeypatch, agent):
    """When ``rounds_used >= max_round`` and we weren't cancelled, the
    AnswerResult should carry ``forced=True`` so callers (OpenAI server)
    can map this to the ``length`` finish_reason.

    Stub ``_core.run_multistep_agent`` to return a synthetic conversation
    with exactly ``max_round`` assistant messages and an empty final answer
    — that's the exact shape of a max_round force-finish.
    """
    max_round = 3
    fake_conv = []
    for i in range(max_round):
        fake_conv.append({"role": "assistant", "content": f"reasoning round {i + 1}"})
        fake_conv.append({"role": "tool", "content": f"result {i + 1}"})

    def fake_run(message, **kwargs):
        return "synthesized answer", fake_conv

    monkeypatch.setattr(agent._core, "run_multistep_agent", fake_run)
    r = agent.answer("test question that is long enough", max_round=max_round)
    assert r.rounds_used == max_round, r.rounds_used
    assert r.forced is True, "should detect max_round force-finish"
    assert r.cancelled is False, "no timeout fired"


def test_answer_result_natural_finish_not_forced(monkeypatch, agent):
    """Natural finish (rounds_used < max_round) leaves forced=False."""
    max_round = 10
    fake_conv = [
        {"role": "assistant", "content": "round 1"},
        {"role": "tool", "content": "tool result"},
        {"role": "assistant", "content": "[FinalAnswer]done"},
    ]

    def fake_run(message, **kwargs):
        return "done", fake_conv

    monkeypatch.setattr(agent._core, "run_multistep_agent", fake_run)
    r = agent.answer("test question that is long enough", max_round=max_round)
    assert r.rounds_used == 2, r.rounds_used
    assert r.forced is False, "natural finish must not set forced"


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
