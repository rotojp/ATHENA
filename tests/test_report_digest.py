# tests/test_report_digest.py
"""Unit tests for the pure report-digest builders (no model/network)."""

from athena_r1._report import (
    REPORT_SYSTEM_PROMPT,
    _humanize_source,
    assemble_digest,
    clean_report,
    digest_from_conversation,
    digest_from_events,
)


def test_assemble_digest_includes_sections_and_sources():
    d = assemble_digest(
        question="Is metformin contraindicated at eGFR 25?",
        short_answer="Yes, contraindicated below eGFR 30.",
        reasoning="Checked renal dosing guidance.",
        tool_findings=[("FDA_lookup", "Contraindicated if eGFR < 30.")],
        char_budget=10_000,
    )
    assert "QUESTION:" in d
    assert "SHORT ANSWER:" in d
    assert "REASONING:" in d
    assert "[Source: FDA Lookup]" in d
    assert "Is metformin contraindicated" in d


def test_assemble_digest_respects_budget_and_truncates():
    huge = "X" * 50_000
    d = assemble_digest(
        question="q",
        short_answer="",
        reasoning="",
        tool_findings=[("BigTool", huge)],
        char_budget=3_000,
    )
    assert len(d) <= 3_000
    assert "[truncated]" in d


def test_assemble_digest_drops_oldest_findings_first():
    # Budget only fits one capped finding (each caps at _PER_ITEM_CAP=2800);
    # the NEWEST must survive.
    findings = [("Old", "A" * 4000), ("New", "B" * 4000)]
    d = assemble_digest("q", "", "", findings, char_budget=3_500)
    assert "[Source: New]" in d
    assert "[Source: Old]" not in d


def test_digest_from_conversation_extracts_pieces():
    conv = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What is the half-life of metformin?"},
        {"role": "assistant", "content": "thinking... [FinalAnswer] About 6 hours."},
        {"role": "tool", "name": "FDA_lookup", "content": "t1/2 ~6h"},
    ]
    d = digest_from_conversation(conv, char_budget=10_000)
    assert "half-life of metformin" in d
    assert "[Source: FDA Lookup]" in d
    assert "About 6 hours." in d  # extracted as SHORT ANSWER


def test_digest_from_events_coalesces_and_pairs_tools():
    events = [
        {"type": "TEXT_MESSAGE_START", "messageId": "m1"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": "Reason "},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": "here."},
        {"type": "TEXT_MESSAGE_END", "messageId": "m1"},
        {"type": "TOOL_CALL_START", "toolCallId": "c1", "toolCallName": "FDA_lookup"},
        {"type": "TOOL_CALL_RESULT", "toolCallId": "c1", "content": "eGFR < 30 → avoid"},
    ]
    d = digest_from_events("the question", "the answer", events, char_budget=10_000)
    assert "the question" in d
    assert "Reason here." in d
    assert "[Source: FDA Lookup] eGFR < 30 → avoid" in d


def test_report_system_prompt_has_four_sections():
    for h in ("Recommendation", "Key Evidence", "Reasoning", "Caveats"):
        assert h in REPORT_SYSTEM_PROMPT


def test_clean_report_strips_think_and_repeats():
    raw = (
        "<think>\ndraft ## Recommendation X ## Caveats Y\n</think>\n"
        "## Recommendation\nReal report.\n## Caveats\nNote.\n"
        "## Recommendation\nDUPLICATE COPY."
    )
    out = clean_report(raw)
    assert "<think>" not in out and "</think>" not in out
    assert out.startswith("## Recommendation")
    assert "Real report." in out
    assert "DUPLICATE COPY" not in out
    assert out.count("## Recommendation") == 1


def test_clean_report_passthrough_single():
    raw = "## Recommendation\nOnly one.\n## Caveats\nDone."
    assert clean_report(raw) == raw.strip()


def test_clean_report_empty():
    assert clean_report("") == ""
    assert clean_report("<think>only thinking</think>   ") == ""


def test_humanize_source_labels():
    assert (
        _humanize_source("FDA_get_dosage_and_storage_information_by_drug_name")
        == "FDA Dosage & Storage"
    )
    assert _humanize_source("FDA_get_pharmacokinetics_by_drug_name") == "FDA Pharmacokinetics"
    assert _humanize_source("Tool_RAG") == "Tool Retrieval"
    assert _humanize_source("") == "Tool"


def test_polish_report_normalizes_headers_and_names():
    from athena_r1._report import _polish_report

    raw = (
        "The max dose is 3200 mg per the "
        "FDA_get_dosage_and_storage_information_by_drug_name tool.\n"
        "Key Evidence:\n- 3200 mg limit.\n"
        "**Reasoning**\nHigher doses raise GI risk.\n"
        "Caveats:\n- Not for renal impairment."
    )
    out = _polish_report(raw)
    assert "FDA_get_" not in out  # raw name humanized away
    assert "FDA Dosage & Storage" in out
    assert out.startswith("## Recommendation")  # leading header inserted
    assert "## Key Evidence" in out
    assert "## Reasoning" in out
    assert "## Caveats" in out


def test_polish_report_empty():
    from athena_r1._report import _polish_report

    assert _polish_report("") == ""
    assert _polish_report("   ") == ""


def test_polish_report_humanizes_invented_bracket_citation():
    from athena_r1._report import _polish_report

    out = _polish_report("## Recommendation\nUse 3200 mg [FDA_get_drug_name_by_dosage_info].")
    assert "FDA_get_" not in out  # raw/invented name humanized
    cited = out.split("[", 1)[1].split("]", 1)[0]
    assert "_" not in cited  # bracket holds a readable label


def test_digest_lists_available_sources():
    d = assemble_digest(
        "q",
        "",
        "",
        [("FDA_get_dosage_and_storage_information_by_drug_name", "x")],
        char_budget=10_000,
    )
    assert "SOURCES AVAILABLE FOR CITATION" in d
    assert "[FDA Dosage & Storage]" in d


def test_valid_source_labels_parses_digest():
    from athena_r1._report import valid_source_labels

    d = assemble_digest(
        "q",
        "",
        "",
        [("FDA_get_pharmacokinetics_by_drug_name", "x"), ("Tool_RAG", "y")],
        char_budget=10_000,
    )
    labels = valid_source_labels(d)
    assert "FDA Pharmacokinetics" in labels
    assert "Tool Retrieval" in labels


def test_polish_strips_hallucinated_citation():
    from athena_r1._report import _polish_report

    raw = (
        "## Recommendation\nUse 1200 mg [FDA Dosage & Storage]. "
        "Risk noted [FDA Warnings]."
    )  # FDA Warnings NOT in trace
    out = _polish_report(raw, valid_sources={"FDA Dosage & Storage"})
    assert "[FDA Dosage & Storage]" in out  # real citation kept
    assert "FDA Warnings" not in out  # hallucinated citation stripped


def test_polish_no_validation_when_sources_unknown():
    from athena_r1._report import _polish_report

    raw = "## Recommendation\nUse 1200 mg [FDA Anything]."
    out = _polish_report(raw, valid_sources=None)
    assert "[FDA Anything]" in out  # not stripped when set unknown
