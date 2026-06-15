# tests/test_report_generate.py
"""generate_report streaming + fallback, with _stream_completion mocked."""

from athena_r1._core import AthenaCore


def _bare_engine():
    # Construct without __init__ so no model/tokenizer is needed.
    eng = AthenaCore.__new__(AthenaCore)
    eng.max_token = 36864
    return eng


def test_generate_report_yields_chunks():
    eng = _bare_engine()
    eng._stream_completion = lambda messages, **kw: iter(["## Recommendation\n", "Use 3200 mg."])
    out = "".join(eng.generate_report("q", "digest"))
    assert "## Recommendation" in out
    assert "Use 3200 mg." in out


def test_generate_report_fallback_on_empty():
    eng = _bare_engine()
    eng._stream_completion = lambda messages, **kw: iter([])
    out = "".join(eng.generate_report("q", "digest"))
    assert out.strip() != ""
    assert "couldn't generate" in out.lower()


def test_report_char_budget_positive_and_scales():
    eng = _bare_engine()
    b = eng.report_char_budget(max_new_tokens=2048)
    assert b > 10_000
    assert b < int(eng.max_token * 3.5)


def test_generate_report_strips_think_block():
    eng = _bare_engine()
    eng._stream_completion = lambda messages, **kw: iter(
        ["<think>\nreasoning ", "here\n</think>\n\n## Recommendation\n", "Do X."]
    )
    out = "".join(eng.generate_report("q", "digest"))
    assert "<think>" not in out
    assert "reasoning here" not in out
    assert out.lstrip().startswith("## Recommendation")
    assert "Do X." in out


def test_generate_report_passes_through_when_no_think():
    eng = _bare_engine()
    eng._stream_completion = lambda messages, **kw: iter(
        ["## Recommendation\n", "Plain report, no think."]
    )
    out = "".join(eng.generate_report("q", "digest"))
    assert out.startswith("## Recommendation")
    assert "Plain report, no think." in out
