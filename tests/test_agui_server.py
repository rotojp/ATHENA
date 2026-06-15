"""End-to-end integration tests for the AG-UI server.

Launches `web/agui_server.py` as a subprocess and POSTs a `RunAgentInput`,
verifying that the SSE response carries valid AG-UI events in the right
order (RUN_STARTED → STEP_STARTED → TEXT_MESSAGE_*/TOOL_CALL_* → RUN_FINISHED).
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
def agui_server(vllm_url, tool_server, model_id):
    """Launch AG-UI server on a free port; teardown after."""
    port = _free_port()
    env = {
        **os.environ,
        "VLLM_URL": vllm_url,
        "TOOLUNIVERSE_API": tool_server,
        "ATHENA_MODEL_PATH": model_id,
        "AZURE_API_KEY": "dummy",
        "ATHENA_R1_LOG_LEVEL": "WARNING",
        "AGUI_PORT": str(port),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(REPO / "src"),
    }
    proc = subprocess.Popen(
        [PYTHON, "-u", str(REPO / "web" / "agui_server.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        # 150s: cold transformers + ag_ui imports off a busy NFS can take
        # 40-60s before uvicorn binds. The old 60s budget sat right at that
        # edge and flaked ("did not become healthy"). On failure, surface the
        # captured subprocess output so a real startup crash is diagnosable
        # instead of a blind timeout.
        deadline = time.time() + 150
        while time.time() < deadline:
            if proc.poll() is not None:  # subprocess exited early — crash
                out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
                pytest.fail(
                    f"agui_server exited during startup (rc={proc.returncode}):\n{out[-2000:]}"
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
            pytest.fail(f"agui_server did not become healthy in 150s:\n{out[-2000:]}")
        yield f"http://0.0.0.0:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _parse_sse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events


def test_health(agui_server):
    r = requests.get(f"{agui_server}/health", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "uptime_s" in body
    # If the agent has been instantiated, its full config should be reported.
    if body.get("initialized"):
        agent = body["agent"]
        for key in ("model", "vllm_url", "tool_server", "max_round"):
            assert key in agent, f"agent.{key} missing from /health"


def test_run_emits_required_events(agui_server):
    """POST a RunAgentInput; verify RUN_STARTED, ≥1 TEXT_MESSAGE, RUN_FINISHED."""
    body = {
        "threadId": "thread-test",
        "runId": "run-test",
        "state": {},
        "messages": [
            {"id": "m1", "role": "user", "content": "Is metformin contraindicated at eGFR 35?"}
        ],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    r = requests.post(
        f"{agui_server}/",
        json=body,
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=300,
    )
    assert r.status_code == 200, r.text
    events = _parse_sse_events(r.text)
    types = [e["type"] for e in events]

    assert "RUN_STARTED" in types, types
    assert "RUN_FINISHED" in types or "RUN_ERROR" in types, types
    # Every successful run should have at least one assistant text message
    # (Stage-1 reasoning is always emitted before final).
    assert "TEXT_MESSAGE_START" in types, types
    assert "TEXT_MESSAGE_END" in types, types
    # Most therapeutic questions trigger at least 1 tool call.
    has_tool = "TOOL_CALL_START" in types
    has_step = "STEP_STARTED" in types
    assert has_tool or has_step, f"expected agent activity, got {set(types)}"


def test_step_events_pair_up(agui_server):
    """Every STEP_STARTED has a matching STEP_FINISHED (AG-UI invariant)."""
    body = {
        "threadId": "thread-step",
        "runId": "run-step",
        "state": {},
        "messages": [{"id": "m1", "role": "user", "content": "What is metformin used for?"}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    r = requests.post(
        f"{agui_server}/",
        json=body,
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=300,
    )
    events = _parse_sse_events(r.text)
    starts = [e for e in events if e["type"] == "STEP_STARTED"]
    finishes = [e for e in events if e["type"] == "STEP_FINISHED"]
    assert len(starts) == len(finishes), (
        f"STEP mismatch: {len(starts)} STARTED vs {len(finishes)} FINISHED. "
        f"start names: {[s.get('stepName') for s in starts]} "
        f"finished names: {[f.get('stepName') for f in finishes]}"
    )


def test_run_ordering_invariants(agui_server):
    """RUN_STARTED appears first, RUN_FINISHED (or RUN_ERROR) last."""
    body = {
        "threadId": "thread-2",
        "runId": "run-2",
        "state": {},
        "messages": [{"id": "m1", "role": "user", "content": "What is metformin used for?"}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    r = requests.post(
        f"{agui_server}/",
        json=body,
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=300,
    )
    events = _parse_sse_events(r.text)
    assert events, "no events received"
    assert events[0]["type"] == "RUN_STARTED"
    assert events[-1]["type"] in ("RUN_FINISHED", "RUN_ERROR")
    # threadId / runId echoed back unchanged
    if events[-1]["type"] == "RUN_FINISHED":
        assert events[-1]["threadId"] == "thread-2"
        assert events[-1]["runId"] == "run-2"


def test_client_abort_does_not_hang_server(agui_server):
    """Client closes SSE mid-stream → server stays responsive on /health
    (and the engine's cancel_event is signaled — see AG-UI server finally block).

    Without the propagation fix, the worker thread would keep spending compute
    on an answer no one will read.
    """
    body = {
        "threadId": "t-abort",
        "runId": "r-abort",
        "state": {},
        "messages": [
            {
                "id": "m1",
                "role": "user",
                "content": (
                    "Give a detailed multi-paragraph dose adjustment writeup for "
                    "metformin in CKD eGFR 35 including drug interactions."
                ),
            }
        ],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    r = requests.post(
        f"{agui_server}/",
        json=body,
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=120,
    )
    assert r.status_code == 200
    events_seen = 0
    for line in r.iter_lines():
        if not line:
            continue
        if line.startswith(b"data:"):
            events_seen += 1
            if events_seen >= 2:
                # We've seen RUN_STARTED + at least one more — yank the connection.
                r.close()
                break
    assert events_seen >= 2, f"only saw {events_seen} events before abort"
    # Server must stay healthy and responsive after the abort.
    time.sleep(2)
    h = requests.get(f"{agui_server}/health", timeout=5)
    assert h.status_code == 200, f"/health unreachable post-abort: {h.status_code}"
    assert h.json()["status"] == "ok"


def test_whitespace_only_input_fast_rejects(agui_server):
    """Whitespace-only user content → RUN_ERROR within ~1s, NOT a wasted
    20s of model compute (regression test for the fold-helper fix)."""
    body = {
        "threadId": "t-ws",
        "runId": "r-ws",
        "state": {},
        "messages": [{"id": "m1", "role": "user", "content": "   \t\n  "}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    t0 = time.time()
    r = requests.post(
        f"{agui_server}/",
        json=body,
        headers={"Accept": "text/event-stream"},
        timeout=30,
    )
    dt = time.time() - t0
    assert r.status_code == 200
    types = [e["type"] for e in _parse_sse_events(r.text)]
    assert "RUN_ERROR" in types, f"expected RUN_ERROR for whitespace, got {types}"
    assert dt < 5.0, f"whitespace took {dt:.1f}s — should be < 5s (no model invocation)"


def test_empty_messages_returns_error_event(agui_server):
    """A request with no user message should emit RUN_ERROR (not crash)."""
    body = {
        "threadId": "thread-3",
        "runId": "run-3",
        "state": {},
        "messages": [],  # nothing for the agent to answer
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    r = requests.post(
        f"{agui_server}/",
        json=body,
        headers={"Accept": "text/event-stream"},
        timeout=30,
    )
    # The endpoint should still 200 — it streams RUN_STARTED + RUN_ERROR.
    events = _parse_sse_events(r.text)
    types = [e["type"] for e in events]
    assert "RUN_ERROR" in types, types


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
