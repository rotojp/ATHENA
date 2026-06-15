"""Tests for AthenaR1 input validation."""

from __future__ import annotations

import sys

import pytest


def test_athena_backend_requires_model_and_vllm_url():
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1, Backend

    with pytest.raises(ValueError, match="Backend.ATHENA requires"):
        AthenaR1(backend=Backend.ATHENA)
    with pytest.raises(ValueError, match="Backend.ATHENA requires"):
        AthenaR1(model="x", backend=Backend.ATHENA)  # missing vllm_url
    with pytest.raises(ValueError, match="Backend.ATHENA requires"):
        AthenaR1(vllm_url="http://x/v1", backend=Backend.ATHENA)  # missing model


def test_gpt_backend_requires_azure_endpoint(monkeypatch):
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1, Backend

    # Clear env so the lazy fallback doesn't satisfy the check.
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="Backend.GPT requires"):
        AthenaR1(backend=Backend.GPT)


def test_gpt_backend_accepts_env_var(monkeypatch):
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1, Backend

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    # Should NOT raise:
    agent = AthenaR1(backend=Backend.GPT)
    assert agent is not None


def test_max_round_must_be_positive():
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1

    with pytest.raises(ValueError, match="max_round must be > 0"):
        AthenaR1(model="x", vllm_url="http://x/v1", max_round=0)
    with pytest.raises(ValueError, match="max_round must be > 0"):
        AthenaR1(model="x", vllm_url="http://x/v1", max_round=-1)


def test_presence_penalty_must_be_in_range():
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1

    with pytest.raises(ValueError, match="presence_penalty must be in"):
        AthenaR1(model="x", vllm_url="http://x/v1", presence_penalty=-0.1)
    with pytest.raises(ValueError, match="presence_penalty must be in"):
        AthenaR1(model="x", vllm_url="http://x/v1", presence_penalty=2.5)


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
