"""API edge-case tests: enum comparison, dataclass repr, map_to_option errors."""

from __future__ import annotations

import sys

import pytest


def test_backend_string_comparison():
    """Backend(str, Enum) allows comparison with raw strings."""
    sys.path.insert(0, "src")
    from athena_r1 import Backend

    assert Backend.ATHENA == "athena"
    assert Backend.GPT == "gpt"
    assert Backend.ATHENA != Backend.GPT
    assert Backend.ATHENA.value == "athena"


def test_round_event_repr_contains_type():
    sys.path.insert(0, "src")
    from athena_r1 import RoundEvent

    ev = RoundEvent(
        type="reasoning",
        content="hello",
        round=2,
        agent_id="main.sub-1",
        agent_level=1,
        parent_agent_id="main",
    )
    r = repr(ev)
    assert "RoundEvent" in r
    assert "reasoning" in r
    assert "main.sub-1" in r


def test_answer_result_repr():
    sys.path.insert(0, "src")
    from athena_r1 import AnswerResult

    r = AnswerResult(answer="hi", rounds_used=3, elapsed_s=12.5, tools_used=["X"])
    rep = repr(r)
    assert "AnswerResult" in rep
    assert "rounds_used=3" in rep
    assert "elapsed_s=12.5" in rep


def test_map_to_option_rejects_unknown_backend(agent):
    """Passing a non-Backend value to map_to_option should raise."""
    with pytest.raises(ValueError, match="Unknown backend"):
        agent.map_to_option(
            conversation=[],
            options={"A": "x", "B": "y"},
            backend="bogus-backend",  # type: ignore[arg-type]
        )


def test_athena_repr_handles_long_model_path():
    sys.path.insert(0, "src")
    from athena_r1 import AthenaR1

    agent = AthenaR1(
        # A locally-downloaded copy of the HF model, passed as a filesystem path.
        model="/data/models/mims-harvard/ATHENA-R1-Qwen3-8B",
        vllm_url="http://localhost:8000/v1",
    )
    r = repr(agent)
    # Just verify nothing crashes + key parts present
    assert "AthenaR1" in r
    assert "ATHENA-R1-Qwen3-8B" in r or "model=" in r


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
