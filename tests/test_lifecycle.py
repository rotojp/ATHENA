"""Tests for AthenaR1 lifecycle: __init__, __repr__, close, context manager."""

from __future__ import annotations

import os
import sys


def test_repr_contains_key_fields():
    """The repr should surface model id, vllm url, max_round, and status."""
    os.environ["TOOLUNIVERSE_API"] = "http://0.0.0.0:8080"
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1

    agent = AthenaR1(
        model="some-model",
        vllm_url="http://hostX:9999/v1",
        max_round=12,
    )
    r = repr(agent)
    assert "AthenaR1" in r
    assert "some-model" in r
    assert "hostX:9999" in r
    assert "max_round=12" in r
    assert "not-initialized" in r


def test_close_resets_initialized_flag():
    os.environ["TOOLUNIVERSE_API"] = "http://0.0.0.0:8080"
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1

    agent = AthenaR1(model="x", vllm_url="http://y/v1")
    agent._initialized = True
    agent.close()
    assert agent._initialized is False


def test_close_drops_lazy_gpt_client(monkeypatch):
    monkeypatch.setenv("TOOLUNIVERSE_API", "http://0.0.0.0:8080")
    monkeypatch.setenv("AZURE_API_KEY", "dummy")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://dummy.openai.azure.com/")
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1

    agent = AthenaR1(model="x", vllm_url="http://y/v1")
    # Force lazy init of the GPT client (instantiates the client object only;
    # no network call happens until it is actually used).
    _ = agent._core.gpt_client
    assert agent._core._gpt_client_obj is not None
    agent.close()
    assert agent._core._gpt_client_obj is None


def test_context_manager_protocol():
    """`with AthenaR1(...) as a:` should call init() on enter, close() on exit."""
    os.environ["TOOLUNIVERSE_API"] = "http://0.0.0.0:8080"
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1

    # Stub init() to avoid loading a real model.
    inited = {"v": False}
    closed = {"v": False}

    class StubbedAgent(AthenaR1):
        def init(self):
            inited["v"] = True
            self._initialized = True

        def close(self):
            closed["v"] = True
            self._initialized = False

    with StubbedAgent(model="x", vllm_url="http://y/v1") as a:
        assert inited["v"] is True
        assert closed["v"] is False
        assert a._initialized is True

    assert closed["v"] is True


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
