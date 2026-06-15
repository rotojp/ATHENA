# tests/test_report_live.py
"""End-to-end report generation against a live vLLM+TU stack. Skipped if down."""

import os
import socket

import pytest

VLLM = os.environ.get("VLLM_URL", "http://0.0.0.0:8000/v1")
TU = os.environ.get("TOOLUNIVERSE_API", "http://0.0.0.0:8080")


def _up(url):
    host, port = url.split("//")[1].split("/")[0].split(":")
    try:
        socket.create_connection((host, int(port)), timeout=1).close()
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _up(VLLM), reason="no live vLLM")


def test_report_from_real_run():
    from athena_r1 import AthenaR1

    a = AthenaR1(model=os.environ["ATHENA_MODEL_PATH"], vllm_url=VLLM, tool_server=TU)
    a.init()
    result = a.answer("Is metformin contraindicated at eGFR 25?")
    report = a.generate_report(result)
    assert report.strip()
    # No think draft, no repetition, and the structured sections present.
    assert "<think>" not in report and "</think>" not in report
    assert report.count("## Recommendation") <= 1, "report must not repeat"
    present = sum(s in report for s in ("Recommendation", "Evidence", "Reasoning", "Caveats"))
    assert present >= 2, f"expected section structure, got: {report[:200]!r}"


def test_report_survives_oversized_trace():
    from athena_r1 import AthenaR1
    from athena_r1._report import digest_from_conversation

    a = AthenaR1(model=os.environ["ATHENA_MODEL_PATH"], vllm_url=VLLM, tool_server=TU)
    a.init()
    conv = [
        {"role": "user", "content": "Is metformin contraindicated at eGFR 25?"},
        {"role": "tool", "name": "FDA_lookup", "content": "X" * 1_000_000},
    ]
    budget = a._core.report_char_budget()
    digest = digest_from_conversation(conv, budget)
    assert len(digest) <= budget
    report = "".join(a._core.generate_report("Is metformin contraindicated at eGFR 25?", digest))
    assert report.strip()
