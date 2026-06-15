"""Tests for AG-UI server's message-folding helper.

Mirrors the corresponding ``_extract_user_question`` tests for the
OpenAI-compat server so both web entry points agree on what context the
agent sees in a multi-round chat.

These are unit tests against the helper directly — no live AG-UI server
needed.
"""

from __future__ import annotations

import sys


def _import_helper():
    """Import the fold helper from web/_messages.py (no AG-UI dep)."""
    sys.path.insert(0, "web")
    from _messages import fold_messages_to_prompt

    # ``_last_user_message`` is the agui_server alias, but it requires
    # the ag-ui-protocol package; ``fold`` is exactly the same function.
    return fold_messages_to_prompt, fold_messages_to_prompt


def test_fold_empty_messages_returns_empty():
    fold, _ = _import_helper()
    assert fold([]) == ""


def test_fold_single_user_message_passes_through():
    fold, _ = _import_helper()
    msgs = [{"id": "1", "role": "user", "content": "What is metformin used for?"}]
    out = fold(msgs)
    # Single-turn: no "Conversation so far" scaffold.
    assert out == "What is metformin used for?"
    assert "Conversation so far" not in out


def test_fold_multi_turn_includes_history():
    fold, _ = _import_helper()
    msgs = [
        {"id": "1", "role": "user", "content": "What is metformin used for?"},
        {"id": "2", "role": "assistant", "content": "Metformin treats type 2 diabetes."},
        {"id": "3", "role": "user", "content": "What's the renal dose adjustment?"},
    ]
    out = fold(msgs)
    assert "Conversation so far:" in out
    assert "user: What is metformin used for?" in out
    assert "assistant: Metformin treats type 2 diabetes." in out
    assert "New user question: What's the renal dose adjustment?" in out


def test_fold_system_message_tagged():
    fold, _ = _import_helper()
    msgs = [
        {"id": "1", "role": "system", "content": "Be a careful clinician."},
        {"id": "2", "role": "user", "content": "First Q"},
        {"id": "3", "role": "assistant", "content": "First A"},
        {"id": "4", "role": "user", "content": "Follow-up Q"},
    ]
    out = fold(msgs)
    assert "[system instruction] Be a careful clinician." in out
    assert "New user question: Follow-up Q" in out


def test_fold_no_user_message_returns_empty():
    """Edge case: a thread with only system + assistant messages is invalid."""
    fold, _ = _import_helper()
    msgs = [
        {"id": "1", "role": "system", "content": "Sys"},
        {"id": "2", "role": "assistant", "content": "Hi"},
    ]
    assert fold(msgs) == ""


def test_fold_multimodal_content_flattens_text_parts():
    """AG-UI multimodal content (list of parts) flattens to text."""
    fold, _ = _import_helper()
    msgs = [
        {
            "id": "1",
            "role": "user",
            "content": [{"text": "First"}, {"text": "Second"}],
        }
    ]
    out = fold(msgs)
    assert "First" in out
    assert "Second" in out


def test_fold_handles_pydantic_like_objects():
    """Folding should also work on objects exposing ``role`` / ``content`` attrs
    (the path actually used by the FastAPI endpoint after AG-UI deserialization).
    """
    fold, _ = _import_helper()

    class _Msg:
        def __init__(self, role: str, content: str) -> None:
            self.role = role
            self.content = content

    msgs = [
        _Msg("user", "Q1"),
        _Msg("assistant", "A1"),
        _Msg("user", "Q2"),
    ]
    out = fold(msgs)
    assert "user: Q1" in out
    assert "assistant: A1" in out
    assert "New user question: Q2" in out


def test_fold_whitespace_only_content_returns_empty():
    """Whitespace-only user content should be treated as no-message, not sent
    to the model (saves ~20s of wasted compute on '   \\n\\t' style inputs)."""
    fold, _ = _import_helper()
    msgs = [{"id": "1", "role": "user", "content": "   \t\n  "}]
    assert fold(msgs) == ""


def test_fold_whitespace_falls_back_to_earlier_user_message():
    """If the LATEST user turn is whitespace-only, treat thread as malformed
    (don't silently fall through to a stale earlier turn)."""
    fold, _ = _import_helper()
    msgs = [
        {"id": "1", "role": "user", "content": "Real first question"},
        {"id": "2", "role": "assistant", "content": "Answer"},
        {"id": "3", "role": "user", "content": "  "},
    ]
    assert fold(msgs) == ""


def test_last_user_message_alias_still_works():
    """Backwards compat: the old _last_user_message export should still resolve."""
    fold, last = _import_helper()
    msgs = [{"id": "1", "role": "user", "content": "hi"}]
    assert last(msgs) == fold(msgs) == "hi"


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
