# src/athena_r1/_report.py
"""Pure helpers for on-demand detailed-report generation.

No model or network access — turns a completed run (a conversation or a
captured AG-UI event trace) into a token-budgeted digest string, and holds
the report system prompt. The budget uses hard truncation so the report can
never re-trigger the token-overflow path that the run pipeline guards against.
"""

from __future__ import annotations

from typing import Any

REPORT_SYSTEM_PROMPT = """You are a clinical evidence summarizer. Using ONLY \
the therapeutic question and the AI agent's reasoning trace and tool findings \
provided, write a thorough, detailed clinical report (aim for ~400-600 words). \
Write all four markdown sections below, each with an H2 header exactly as named, \
at the depth shown in the example.

## Recommendation
The bottom-line answer ONLY, in 1-3 sentences: the overall conclusion plus the \
single most decision-relevant number or threshold. This is a high-level takeaway \
— do NOT list the supporting specifics here; those belong in Key Evidence. Never \
restate facts that you also put in Key Evidence. For a multi-part question (e.g. \
"across pregnancy, geriatric, pediatric, and renal"), still write ONE synthesised \
paragraph capturing the overall picture and the single most important caution — \
do NOT enumerate each part with its own numbers here (the per-part detail goes in \
Key Evidence, one bullet per part).

## Key Evidence
6-10 bullet points — the substantive core of the report. Start each bullet with \
a short **bold lead-in label** naming what the bullet is about, then the \
specific finding with exact numbers, values, doses, and conditions from the \
trace. End each bullet with the source(s) in brackets using the readable label \
exactly as it appears in the trace, e.g. "[FDA Dosage & Storage]". If a finding \
draws on more than one source, cite all of them, e.g. "[FDA Warnings][FDA \
Pharmacokinetics]". Draw on EVERY relevant tool finding and cite the different \
sources that actually contributed — do not lean on a single source.

## Reasoning
2-3 paragraphs (or a short paragraph plus bullets) explaining HOW the evidence \
supports the recommendation: the clinical rationale and mechanism (why a \
threshold exists, what risk it mitigates, how findings interrelate), and how \
conflicting or graded evidence is weighed. Cite sources inline here too where a \
claim rests on a specific finding.

## Caveats
4-6 specific bullet points: populations or scenarios not covered, assumptions \
made, drug interactions or comorbidities that could change the answer, anything \
that could not be verified from the trace, and when a clinician should be \
consulted. Be concrete, not generic.

Follow this exact format, depth, and style (illustrative content only — use the \
real trace):

## Recommendation
Metformin is contraindicated at an eGFR of 25 and should not be used; the \
governing threshold is an eGFR of 30 mL/min/1.73 m². If renal function is \
borderline, reassess before initiating and prefer an alternative agent.

## Key Evidence
- **Contraindication threshold:** Metformin is contraindicated when eGFR is below 30 mL/min/1.73 m² [FDA Contraindications].
- **Caution range:** Initiation is not recommended between an eGFR of 30 and 45; therapy already in that range needs dose reduction and closer monitoring [FDA Dosage & Administration].
- **Limiting toxicity:** The dose-limiting harm is metformin-associated lactic acidosis, whose risk rises steeply as renal clearance falls [FDA Warnings][FDA Pharmacokinetics].
- **Pharmacokinetic basis:** Metformin is cleared largely unchanged by the kidney, so reduced clearance drives drug accumulation [FDA Pharmacokinetics].
- **Aggravating factors:** Dehydration, acute kidney injury, and iodinated contrast transiently worsen clearance and can precipitate toxicity even above the threshold [FDA Warnings].

## Reasoning
Metformin is renally excreted, so impaired kidneys allow it to accumulate and
promote lactic acidosis [FDA Pharmacokinetics]. The eGFR-30 threshold marks the
point where that risk outweighs the glycemic benefit, which is why use below it
is contraindicated outright rather than merely dose-adjusted [FDA
Contraindications].

Between an eGFR of 30 and 45 the picture is graded rather than binary: the drug
may be continued with a reduced dose and tighter monitoring, but new starts are
discouraged [FDA Dosage & Administration]. This reflects a risk that scales with
declining clearance rather than switching on at a single cutoff.

## Caveats
- Applies to standard adult dosing; does not cover dialysis patients or pediatric use.
- Acute illness, dehydration, or contrast media can transiently push renal function below the safe threshold.
- Concurrent nephrotoxic drugs can raise the effective risk at a higher eGFR.
- eGFR should be confirmed and trended, not read from a single value, before a decision.

Rules: Begin directly with "## Recommendation" (no preamble and do not restate \
the question). Refer to sources ONLY by their bracketed readable label; never \
write a raw function name such as "FDA_get_dosage_..._by_drug_name". CRITICAL: \
the citation labels in the example above (e.g. [FDA Contraindications], [FDA \
Warnings]) are ILLUSTRATIVE ONLY — you must cite ONLY the labels listed under \
"SOURCES AVAILABLE FOR CITATION" in the provided trace, and never any other \
label even if it appeared in the example. Do not invent facts or sources. Be \
precise, specific, and clinically careful; prefer concrete facts and numbers \
over vague statements."""

# Rough chars-per-token for budgeting without invoking the tokenizer.
_CHARS_PER_TOKEN = 3.5
# Per-finding cap so one huge tool output cannot dominate the digest, but
# generous enough that the model has real material to write a detailed report.
_PER_ITEM_CAP = 2800


def _humanize_source(name: str) -> str:
    """Turn a raw tool name into a readable citation label.

    e.g. ``FDA_get_dosage_and_storage_information_by_drug_name`` →
    ``FDA Dosage & Storage``; ``Tool_RAG`` → ``Tool Retrieval``. Keeps a short
    all-caps provider token (FDA, EMA) intact and title-cases the rest, so
    report citations read cleanly instead of echoing the function name.
    """
    s = str(name or "tool").strip()
    special = {
        "Tool_RAG": "Tool Retrieval",
        "Finish": "Finish",
        "CallAgent": "Sub-agent",
        "tool": "Tool",
    }
    if s in special:
        return special[s]
    for suf in ("_by_drug_name", "_by_name", "_information", "_info"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = s.replace("_get_", "_")
    if s.startswith("get_"):
        s = s[4:]
    out = []
    for p in s.split("_"):
        if not p:
            continue
        if p.isupper() and len(p) <= 4:
            out.append(p)  # provider acronym (FDA, EMA, NIH…)
        elif p.lower() == "and":
            out.append("&")
        else:
            out.append(p.capitalize())
    return " ".join(out) or "Tool"


def assemble_digest(
    question: str,
    short_answer: str,
    reasoning: str,
    tool_findings: list[tuple[str, str]],
    char_budget: int,
) -> str:
    """Build the digest string, keeping question/answer/reasoning in full and
    fitting as many (capped) tool findings as the budget allows. Newest
    findings are kept first; oldest are dropped when the budget is exhausted.
    The returned string never exceeds ``char_budget``.
    """
    head_parts = [f"QUESTION:\n{(question or '').strip()}\n"]
    if short_answer and short_answer.strip():
        head_parts.append(f"SHORT ANSWER:\n{short_answer.strip()}\n")
    if reasoning and reasoning.strip():
        head_parts.append(f"REASONING:\n{reasoning.strip()}\n")
    head = "\n".join(head_parts)

    remaining = char_budget - len(head) - len("\nTOOL FINDINGS:\n") - 8
    rendered: list[str] = []
    used_labels: list[str] = []
    for source, content in reversed(tool_findings):  # newest first
        if remaining <= 0:
            break
        c = str(content or "").strip()
        if len(c) > _PER_ITEM_CAP:
            c = c[:_PER_ITEM_CAP] + " …[truncated]"
        label = _humanize_source(source)
        block = f"[Source: {label}] {c}"
        if len(block) + 1 > remaining:
            continue
        rendered.append(block)
        if label not in used_labels:
            used_labels.append(label)
        remaining -= len(block) + 1
    rendered.reverse()  # back to chronological order

    out = head
    if rendered:
        # An explicit allow-list of the ONLY labels the report may cite. This
        # is what stops the model from copying citation labels out of the
        # few-shot example in the system prompt (sources that aren't in this
        # run's trace).
        srcline = "SOURCES AVAILABLE FOR CITATION (cite ONLY these): " + "; ".join(
            f"[{lab}]" for lab in used_labels
        )
        out = head + "\n" + srcline + "\n\nTOOL FINDINGS:\n" + "\n".join(rendered)
    out = out.strip()
    return out[:char_budget]


def valid_source_labels(digest: str) -> set[str]:
    """The set of citation labels a report built from ``digest`` may use —
    parsed from the digest's TOOL FINDINGS ``[Source: X]`` markers. Used to
    strip hallucinated citations (e.g. labels copied from the prompt example)."""
    import re

    return set(re.findall(r"\[Source:\s*([^\]]+?)\s*\]", digest or ""))


def _extract_short_answer(assistant_text: str) -> str:
    txt = assistant_text or ""
    if "[FinalAnswer]" in txt:
        txt = txt.split("[FinalAnswer]")[-1]
    if "</think>" in txt:
        txt = txt.split("</think>")[-1]
    return txt.strip()


def digest_from_conversation(conversation: list[dict[str, Any]], char_budget: int) -> str:
    """Digest from a run's conversation (Python API / OpenAI path)."""
    question = ""
    reasoning_parts: list[str] = []
    short_answer = ""
    tool_findings: list[tuple[str, str]] = []
    for m in conversation or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        content = str(m.get("content", "") or "")
        if role == "user" and not question:
            question = content
        elif role == "assistant":
            if content.strip():
                reasoning_parts.append(content.strip())
            sa = _extract_short_answer(content)
            if sa:
                short_answer = sa
        elif role in ("tool", "function"):
            tool_findings.append((m.get("name") or "tool", content))
    reasoning = "\n\n".join(reasoning_parts)
    return assemble_digest(question, short_answer, reasoning, tool_findings, char_budget)


def digest_from_events(
    question: str, short_answer: str, events: list[dict[str, Any]], char_budget: int
) -> str:
    """Digest from a captured AG-UI event trace (web path). ``question`` and
    ``short_answer`` are passed in (they are not part of the event stream)."""
    msg_buf: dict[str, str] = {}
    reasoning_parts: list[str] = []
    tool_names: dict[str, str] = {}
    tool_findings: list[tuple[str, str]] = []
    for e in events or []:
        if not isinstance(e, dict):
            continue
        t = e.get("type")
        if t == "TEXT_MESSAGE_START":
            msg_buf[e.get("messageId")] = ""
        elif t == "TEXT_MESSAGE_CONTENT":
            mid = e.get("messageId")
            if mid in msg_buf:
                msg_buf[mid] += e.get("delta", "") or ""
        elif t == "TEXT_MESSAGE_END":
            txt = (msg_buf.pop(e.get("messageId"), "") or "").strip()
            if txt:
                reasoning_parts.append(txt)
        elif t == "TOOL_CALL_START":
            tool_names[e.get("toolCallId")] = e.get("toolCallName") or "tool"
        elif t == "TOOL_CALL_RESULT":
            content = e.get("content", "") or ""
            if str(content).strip():
                src = tool_names.get(e.get("toolCallId"), "tool")
                tool_findings.append((src, content))
    reasoning = "\n\n".join(reasoning_parts)
    return assemble_digest(question, short_answer, reasoning, tool_findings, char_budget)


def clean_report(raw: str, valid_sources: set[str] | None = None) -> str:
    """Extract the first clean report copy from raw model output.

    The agent-trained model drafts the report inside a <think>...</think> block
    and tends to re-emit the report several times. Drop the think draft, then
    keep only the first complete report — cut at a repeated "## Recommendation"
    or any later "</think>". Returns the stripped report text (possibly empty).

    ``valid_sources`` (the citation labels actually present in the run's trace,
    from ``valid_source_labels(digest)``) lets the polish step strip
    hallucinated citations — labels the model copied from the prompt's few-shot
    example that aren't in this run.
    """
    text = raw or ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.replace("<think>", "")
    cut = None
    first = text.find("## Recommendation")
    if first != -1:
        second = text.find("## Recommendation", first + len("## Recommendation"))
        if second != -1:
            cut = second
    tclose = text.find("</think>")
    if tclose != -1:
        cut = tclose if cut is None else min(cut, tclose)
    if cut is not None:
        text = text[:cut]
    return _polish_report(text, valid_sources)


_SECTION_NAMES = ("Recommendation", "Key Evidence", "Reasoning", "Caveats")


def _polish_report(text: str, valid_sources: set[str] | None = None) -> str:
    """Normalize a generated report so formatting is reliable regardless of the
    (out-of-distribution) model's inconsistencies:
      * humanize any raw tool/function name that leaked into the prose,
      * coerce section-header variants ("**Reasoning**", "Reasoning:",
        "### Reasoning", a bare "Reasoning" line) into "## Reasoning",
      * ensure the report opens with "## Recommendation",
      * if ``valid_sources`` is given, strip hallucinated citations (labels
        not in the run's trace — e.g. copied from the prompt example).
    Empty input stays empty.
    """
    import re

    t = (text or "").strip()
    if not t:
        return ""
    # Raw function names that slipped past the prompt → readable labels.
    # Covers any "<provider>_get_..." token in prose, plus any bracketed
    # citation still containing an underscore (a raw or invented function
    # name), so no snake_case function name is ever shown to the user.
    t = re.sub(r"\b[A-Za-z][A-Za-z]*_get_[A-Za-z_]+\b", lambda m: _humanize_source(m.group(0)), t)
    t = re.sub(r"\bTool_RAG\b", "Tool Retrieval", t)
    t = re.sub(r"\[([^\]\n]*_[^\]\n]*)\]", lambda m: "[" + _humanize_source(m.group(1)) + "]", t)
    # Strip hallucinated citations: any [label] that isn't a real trace source.
    # Only applied when we know the valid set AND it's non-empty (otherwise we
    # can't distinguish a real citation from a fabricated one).
    if valid_sources:

        def _check(m):
            return m.group(0) if m.group(1).strip() in valid_sources else ""

        t = re.sub(r"\[([^\]\n]+)\]", _check, t)
        # Tidy any " ." / " []" left by a removed trailing citation.
        t = re.sub(r"[ \t]+([.;,])", r"\1", t)
        t = re.sub(r"[ \t]{2,}", " ", t)
    # Header normalization: a line that is ONLY a section name (optionally with
    # #, * or a trailing colon) becomes a clean "## Name".
    for name in _SECTION_NAMES:
        t = re.sub(
            rf"(?im)^[ \t]*#{{0,6}}[ \t]*\*{{0,2}}{re.escape(name)}\*{{0,2}}[ \t]*:?[ \t]*$",
            f"## {name}",
            t,
        )
    if "## Recommendation" not in t:
        t = "## Recommendation\n" + t
    return t.strip()
