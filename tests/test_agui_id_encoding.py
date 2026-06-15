"""Unit tests for the agent_id encoding inside AG-UI event IDs.

The AG-UI standard's TextMessage / ToolCall events don't carry an explicit
``agent_id``, but their ``message_id`` / ``tool_call_id`` are opaque strings
so we encode the emitting agent_id into the ID itself. This lets the bundled
HTML demo (and any other AG-UI client) attribute events to the right
sub-agent swimlane without the "attach to most-recent step" heuristic.
"""

from __future__ import annotations

import sys


def _import_helpers():
    sys.path.insert(0, "web")
    from agui_server import _msg_id, _tool_call_id, agent_id_from_event_id

    return _msg_id, _tool_call_id, agent_id_from_event_id


def test_msg_id_encodes_agent_id():
    msg_id_fn, _, decode = _import_helpers()
    mid = msg_id_fn("main")
    assert decode(mid) == "main"


def test_msg_id_handles_nested_agent_id():
    msg_id_fn, _, decode = _import_helpers()
    mid = msg_id_fn("main.sub-1")
    assert decode(mid) == "main.sub-1"


def test_msg_id_handles_deeply_nested_agent_id():
    msg_id_fn, _, decode = _import_helpers()
    mid = msg_id_fn("main.sub-1.sub-2.sub-3")
    assert decode(mid) == "main.sub-1.sub-2.sub-3"


def test_tool_call_id_encodes_agent_id():
    _, tc_fn, decode = _import_helpers()
    tid = tc_fn("main.sub-1")
    assert decode(tid) == "main.sub-1"


def test_legacy_id_falls_back_to_main():
    """Old msg-xxxx style IDs (no separator) should decode to "main"."""
    _, _, decode = _import_helpers()
    assert decode("msg-abc123") == "main"
    assert decode("tc-deadbeef") == "main"


def test_decode_none_or_empty_safe():
    _, _, decode = _import_helpers()
    assert decode(None) == "main"
    assert decode("") == "main"


def test_msg_ids_are_unique_per_call():
    msg_id_fn, _, _ = _import_helpers()
    ids = {msg_id_fn("main") for _ in range(100)}
    assert len(ids) == 100, "msg_id collisions"


def test_tool_call_ids_are_unique_per_call():
    _, tc_fn, _ = _import_helpers()
    ids = {tc_fn("main.sub-1") for _ in range(100)}
    assert len(ids) == 100, "tool_call_id collisions"


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
