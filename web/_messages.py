"""Message-folding helpers shared by the AG-UI and OpenAI-compat servers.

ATHENA-R1's Stage-1 engine takes a single ``message`` string. To support
multi-round chat clients (Open WebUI, agent-chat-ui, CopilotKit, the bundled
demo, etc.) we fold the full conversation into one labelled transcript:

    Conversation so far:
    [system instruction] ...
    user: ...
    assistant: ...

    New user question: <latest user message>

This module deliberately has **no AG-UI / OpenAI dependencies** so it can be
unit-tested without pulling in the optional web extras.
"""

from __future__ import annotations

from typing import Any


def _msg_field(m: Any, name: str) -> Any:
    """Read ``name`` off a message that may be a Pydantic model or a plain dict."""
    return getattr(m, name, None) or (m.get(name) if isinstance(m, dict) else None)


def _content_to_text(content: Any) -> str:
    """Flatten message ``content`` (str | list of multimodal parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            t
            for part in content
            if (
                t := getattr(part, "text", None)
                or (part.get("text") if isinstance(part, dict) else None)
            )
        )
    return ""


def fold_messages_to_prompt(messages: list) -> str:
    """Build the Stage-1 ``question`` string from a chat-style messages list.

    Returns "" if no user message is present so the caller can emit a 4xx
    error / RUN_ERROR uniformly.
    """
    if not messages:
        return ""

    # Locate the LAST user-role message. If it's missing or whitespace-only,
    # treat the thread as malformed — do NOT silently fall back to an older
    # user turn, since that would have the agent answer a stale question.
    # We record the *index* so the history slice can exclude exactly this
    # message; using ``messages[:-1]`` would wrongly include the latest user
    # message in "Conversation so far" whenever the client appends a
    # trailing assistant/tool turn after the user (which agent-chat-ui and
    # CopilotKit both do during retries).
    last_user_text = ""
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if _msg_field(messages[i], "role") == "user":
            last_user_text = _content_to_text(_msg_field(messages[i], "content"))
            last_user_idx = i
            break
    if not last_user_text or not last_user_text.strip():
        return ""

    if len(messages) <= 1:
        return last_user_text

    history_lines: list[str] = []
    for m in messages[:last_user_idx]:
        role = _msg_field(m, "role")
        text = _content_to_text(_msg_field(m, "content"))
        if not text:
            continue
        if role == "system":
            history_lines.append(f"[system instruction] {text}")
        elif role in ("user", "assistant"):
            history_lines.append(f"{role}: {text}")
    if not history_lines:
        return last_user_text

    return (
        "Conversation so far:\n"
        + "\n".join(history_lines)
        + "\n\nNew user question: "
        + last_user_text
    )
