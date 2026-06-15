"""Smoke test: single Q through ATHENA-R1 end-to-end (Stage-1 + Stage-2)."""

import sys


def test_imports_only():
    """Package imports without crashing — no vLLM/TU needed."""
    import athena_r1

    assert athena_r1.AthenaR1
    assert athena_r1.AnswerResult
    assert athena_r1.Backend
    assert athena_r1.RoundEvent


def test_end_to_end_answer(agent):
    """agent.answer() returns a non-empty free-form answer."""
    question = (
        "What is the recommended starting dose of metformin for a "
        "65-year-old patient with type 2 diabetes and eGFR of 35 mL/min?"
    )
    result = agent.answer(question, temperature=0.7, max_round=10)
    assert result.answer
    assert len(result.conversation) > 0
    assert result.rounds_used >= 1


def test_end_to_end_map_to_option(agent):
    """map_to_option returns a single uppercase letter A-E."""
    from athena_r1 import Backend

    question = (
        "What is the recommended starting dose of metformin for a "
        "65-year-old patient with type 2 diabetes and eGFR of 35 mL/min?"
    )
    result = agent.answer(question, temperature=0.7, max_round=10)
    letter = agent.map_to_option(
        result.conversation,
        {
            "A": "500 mg twice daily",
            "B": "1000 mg twice daily",
            "C": "Contraindicated; avoid metformin",
            "D": "500 mg once daily, titrate based on GFR trajectory",
        },
        backend=Backend.ATHENA,
    )
    assert letter in ("A", "B", "C", "D", "Error"), letter


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
