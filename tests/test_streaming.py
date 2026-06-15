"""Streaming integration test: verify answer_streaming() yields the expected event types."""

import sys


def test_streaming_event_types(agent):
    """answer_streaming yields at least one of each expected event type
    and finishes with final_answer.
    """
    question = (
        "What is the recommended starting dose of metformin for a "
        "65-year-old patient with type 2 diabetes and eGFR of 35 mL/min?"
    )
    counts: dict[str, int] = {}
    final_event = None
    for ev in agent.answer_streaming(question, temperature=0.7, max_round=10):
        counts[ev.type] = counts.get(ev.type, 0) + 1
        if ev.type == "final_answer":
            final_event = ev

    assert counts.get("round_start", 0) >= 1, counts
    assert counts.get("reasoning", 0) >= 1, counts
    assert counts.get("final_answer", 0) == 1, counts
    assert final_event is not None
    assert final_event.content


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
