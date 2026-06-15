"""Utility helpers used by the multi-step reasoning engine."""

import copy
import json


def convert_to_qwen_format(messages: list) -> list:
    """Normalize tool-call messages into Qwen3 chat-template expectations.

    The Qwen3 template expects ``tool_calls`` to be a list of dicts, but
    upstream callers sometimes pass them as a JSON-encoded string. Deep-copy
    + parse so we don't mutate the caller's data.
    """
    messages = copy.deepcopy(messages)
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            if isinstance(msg["tool_calls"], str):
                msg["tool_calls"] = json.loads(msg["tool_calls"])
    return messages
