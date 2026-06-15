# tests/test_report_api.py
"""AthenaR1.generate_report wires conversation → digest → core, mocked."""

from athena_r1.agent import AnswerResult, AthenaR1


def _bare_agent():
    a = AthenaR1.__new__(AthenaR1)

    class _Core:
        max_token = 36864

        def report_char_budget(self, max_new_tokens=2048):
            return 100_000

        def generate_report(self, question, digest, **kw):
            assert "half-life" in question  # extracted from conversation
            assert "[Source: FDA Lookup]" in digest
            yield "REPORT BODY"

    a._core = _Core()
    return a


def _result():
    return AnswerResult(
        answer="About 6 hours.",
        conversation=[
            {"role": "user", "content": "What is the half-life of metformin?"},
            {"role": "assistant", "content": "[FinalAnswer] About 6 hours."},
            {"role": "tool", "name": "FDA_lookup", "content": "t1/2 ~6h"},
        ],
    )


def test_generate_report_returns_joined_string():
    a = _bare_agent()
    out = a.generate_report(_result())
    assert out == "REPORT BODY"


def test_generate_report_stream_yields():
    a = _bare_agent()
    chunks = list(a.generate_report(_result(), stream=True))
    assert "".join(chunks) == "REPORT BODY"
