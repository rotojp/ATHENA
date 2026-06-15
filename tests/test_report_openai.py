# tests/test_report_openai.py
"""report:true returns the report instead of the short answer; default unchanged."""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "web")
import importlib

from fastapi.testclient import TestClient


def _client(monkeypatch):
    mod = importlib.import_module("openai_server")
    from athena_r1.agent import AnswerResult

    class _Agent:
        def answer(self, question, **kw):
            return AnswerResult(
                answer="Short.",
                conversation=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "[FinalAnswer] Short."},
                ],
            )

        def generate_report(self, result_or_conversation, *, stream=False, **kw):
            return "## Recommendation\nLong report."

    monkeypatch.setattr(mod, "get_agent", lambda: _Agent())
    return TestClient(mod.app)


def test_report_flag_returns_report(monkeypatch):
    client = _client(monkeypatch)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "athena",
            "stream": False,
            "report": True,
            "messages": [{"role": "user", "content": "metformin renal dosing?"}],
        },
    )
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert "Long report." in content


def test_default_returns_short_answer(monkeypatch):
    client = _client(monkeypatch)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "athena",
            "stream": False,
            "messages": [{"role": "user", "content": "metformin renal dosing?"}],
        },
    )
    content = r.json()["choices"][0]["message"]["content"]
    assert "Short." in content


def test_report_with_stream_is_rejected(monkeypatch):
    client = _client(monkeypatch)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "athena",
            "stream": True,
            "report": True,
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 400
