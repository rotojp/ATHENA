# tests/test_report_endpoint.py
"""POST /report streams a report; agent._core.generate_report mocked."""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "web")
import importlib

from fastapi.testclient import TestClient


def _client(monkeypatch):
    mod = importlib.import_module("agui_server")

    class _Core:
        def report_char_budget(self, max_new_tokens=2048):
            return 100_000

        def report_snapshots(self, question, digest, **kw):
            assert question == "the q"
            # Progressive full-report snapshots (replace semantics).
            yield "## Recommendation\nDo "
            yield "## Recommendation\nDo X."

    class _Agent:
        _core = _Core()

    monkeypatch.setattr(mod, "get_agent", lambda: _Agent())
    return TestClient(mod.app)


def test_report_endpoint_streams(monkeypatch):
    client = _client(monkeypatch)
    body = {
        "question": "the q",
        "final_answer": "X",
        "events": [
            {"type": "TOOL_CALL_START", "toolCallId": "c1", "toolCallName": "FDA_lookup"},
            {"type": "TOOL_CALL_RESULT", "toolCallId": "c1", "content": "evidence"},
        ],
    }
    with client.stream("POST", "/report", json=body) as r:
        assert r.status_code == 200
        text = "".join(chunk for chunk in r.iter_text())
    assert "## Recommendation" in text
    assert "Do X." in text


def test_report_endpoint_400_without_question(monkeypatch):
    client = _client(monkeypatch)
    r = client.post("/report", json={"question": "", "events": []})
    assert r.status_code == 400
