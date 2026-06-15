"""Tests for agent.info() introspection."""

from __future__ import annotations

import os
import sys


def test_info_returns_expected_keys():
    os.environ["TOOLUNIVERSE_API"] = "http://0.0.0.0:8080"
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1

    agent = AthenaR1(
        model="some-model",
        vllm_url="http://x/v1",
        max_round=15,
        max_agent_level=2,
    )
    info = agent.info()
    expected_keys = {
        "model",
        "vllm_url",
        "tool_server",
        "backend",
        "max_round",
        "max_token",
        "max_agent_level",
        "init_rag_num",
        "step_rag_num",
        "cache_tool",
        "enable_rag",
        "tool_categories",
        "tools_loaded",
        "initialized",
    }
    assert expected_keys.issubset(info.keys())
    assert info["model"] == "some-model"
    assert info["vllm_url"] == "http://x/v1"
    assert info["max_round"] == 15
    assert info["max_agent_level"] == 2
    assert info["initialized"] is False
    assert isinstance(info["tool_categories"], list)
    assert len(info["tool_categories"]) >= 1


def test_info_is_json_serializable():
    """The dict should round-trip through json without errors."""
    import json

    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1

    agent = AthenaR1(model="x", vllm_url="http://y/v1")
    info = agent.info()
    blob = json.dumps(info)
    parsed = json.loads(blob)
    assert parsed == info


def test_info_works_before_init():
    """info() must not require init() — useful for /health endpoints."""
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1

    agent = AthenaR1(model="x", vllm_url="http://y/v1")
    assert agent._initialized is False
    # Should NOT raise:
    info = agent.info()
    assert info["initialized"] is False
    assert info["tools_loaded"] == 0


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
