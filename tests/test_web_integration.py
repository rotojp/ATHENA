"""End-to-end web UI integration tests.

Tests that actually hit the live OpenAI-compat HTTP server (started as a
background process by the test fixture). Skipped if no vLLM/TU server is
running. The AG-UI server is covered separately by ``test_agui_server.py``.

Run:
    PYTHONPATH=src pytest tests/test_web_integration.py -v -s
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def openai_server(vllm_url, tool_server, model_id):
    """Launch the OpenAI-compat server on a free port; tear down after."""
    port = _free_port()
    env = {
        **os.environ,
        "VLLM_URL": vllm_url,
        "TOOLUNIVERSE_API": tool_server,
        "ATHENA_MODEL_PATH": model_id,
        "AZURE_API_KEY": "dummy",
        "ATHENA_R1_LOG_LEVEL": "WARNING",
        "PORT": str(port),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(REPO / "src"),
    }
    proc = subprocess.Popen(
        [PYTHON, "-u", str(REPO / "web" / "openai_server.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        # 150s: cold transformers imports off a busy NFS can take 40-60s
        # before uvicorn binds; the old 60s budget flaked. On failure, surface
        # the captured subprocess output so a real crash is diagnosable.
        deadline = time.time() + 150
        while time.time() < deadline:
            if proc.poll() is not None:  # subprocess exited early — crash
                out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
                pytest.fail(
                    f"openai_server exited during startup (rc={proc.returncode}):\n{out[-2000:]}"
                )
            try:
                r = requests.get(f"http://0.0.0.0:{port}/health", timeout=3)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            proc.kill()
            out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
            pytest.fail(f"openai_server did not become healthy in 150s:\n{out[-2000:]}")
        yield f"http://0.0.0.0:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_health_endpoint(openai_server):
    """GET /health returns OK + advertises the model id."""
    r = requests.get(f"{openai_server}/health", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model" in body
    assert "version" in body
    assert "uptime_s" in body
    # If the agent is already up, its full config should be returned.
    if body.get("initialized"):
        agent = body["agent"]
        for key in ("model", "vllm_url", "tool_server", "max_round"):
            assert key in agent, f"agent.{key} missing from /health"


def test_list_models(openai_server):
    """GET /v1/models returns OpenAI-format list with one entry."""
    r = requests.get(f"{openai_server}/v1/models", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) >= 1
    assert body["data"][0]["object"] == "model"


def test_chat_completions_nonstreaming(openai_server):
    """POST /v1/chat/completions (stream=false) returns full response."""
    r = requests.post(
        f"{openai_server}/v1/chat/completions",
        json={
            "model": "athena-r1",
            "messages": [{"role": "user", "content": "Is metformin contraindicated at eGFR 35?"}],
            "temperature": 0.7,
            "stream": False,
        },
        timeout=300,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert len(body["choices"]) == 1
    content = body["choices"][0]["message"]["content"]
    assert content
    assert len(content) > 20, f"answer suspiciously short: {content!r}"


def test_chat_completions_streaming(openai_server):
    """POST /v1/chat/completions (stream=true) returns valid SSE chunks."""
    r = requests.post(
        f"{openai_server}/v1/chat/completions",
        json={
            "model": "athena-r1",
            "messages": [{"role": "user", "content": "Is metformin contraindicated at eGFR 35?"}],
            "temperature": 0.7,
            "stream": True,
        },
        stream=True,
        timeout=300,
    )
    assert r.status_code == 200

    chunks = []
    saw_final = False
    for line in r.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line.startswith("data: "):
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            chunks.append(obj)
            # Look for finish_reason == stop in any chunk.
            for choice in obj.get("choices", []):
                if choice.get("finish_reason") == "stop":
                    saw_final = True
                    break

    assert len(chunks) >= 5, f"too few SSE chunks: {len(chunks)}"
    assert saw_final, "never saw finish_reason=stop"
    # Verify every chunk has OpenAI-format keys.
    for c in chunks:
        assert c["object"] == "chat.completion.chunk"
        assert "choices" in c
        assert "id" in c
        assert "created" in c


def test_streaming_includes_round_markers(openai_server):
    """SSE stream should include at least one [round N] marker."""
    r = requests.post(
        f"{openai_server}/v1/chat/completions",
        json={
            "model": "athena-r1",
            "messages": [{"role": "user", "content": "What are Kisunla FDA contraindications?"}],
            "temperature": 0.7,
            "stream": True,
        },
        stream=True,
        timeout=300,
    )
    assert r.status_code == 200
    all_content = []
    for line in r.iter_lines():
        if not line:
            continue
        s = line.decode("utf-8")
        if not s.startswith("data: "):
            continue
        if s[6:].strip() == "[DONE]":
            break
        try:
            obj = json.loads(s[6:])
        except json.JSONDecodeError:
            continue
        for choice in obj.get("choices", []):
            delta = choice.get("delta", {})
            if "content" in delta and delta["content"]:
                all_content.append(delta["content"])
    full = "".join(all_content)
    assert "[round" in full, "stream missing round markers"
    assert "[final]" in full or "[tool_call" in full or "[tool_result" in full, full[:500]


def test_empty_question_rejected(openai_server):
    """A POST with no user messages must 400."""
    r = requests.post(
        f"{openai_server}/v1/chat/completions",
        json={
            "model": "athena-r1",
            "messages": [{"role": "system", "content": "be helpful"}],
            "stream": False,
        },
        timeout=30,
    )
    assert r.status_code == 400
    assert "no user message" in r.json()["detail"].lower()


def test_empty_messages_rejected(openai_server):
    """A POST with empty messages must 400."""
    r = requests.post(
        f"{openai_server}/v1/chat/completions",
        json={"model": "athena-r1", "messages": [], "stream": False},
        timeout=30,
    )
    assert r.status_code == 400


def test_short_question_handled(openai_server):
    """Very short (<=10 char) question gets graceful error string, not crash."""
    r = requests.post(
        f"{openai_server}/v1/chat/completions",
        json={
            "model": "athena-r1",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
        timeout=60,
    )
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert "valid message" in content.lower() or len(content) > 10


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v", "-s"]))
