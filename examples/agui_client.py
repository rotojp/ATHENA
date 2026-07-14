"""Python client for the AG-UI server.

Demonstrates how to consume the AG-UI SSE stream from any Python script,
without a React frontend. Useful for paper figures, automated regression
tests, or integration into LangGraph-style supervisors.

Prerequisites:
    1. AG-UI server running:  python web/agui_server.py
"""

import json
import os
import sys

import requests

AGUI_URL = os.environ.get("AGUI_URL", "http://127.0.0.1:8090")


def run_question(question: str) -> None:
    body = {
        "threadId": "demo-thread",
        "runId": "demo-run",
        "state": {},
        "messages": [{"id": "m1", "role": "user", "content": question}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    print(f"POST {AGUI_URL}/ — {question[:60]}...\n")
    with requests.post(
        f"{AGUI_URL}/",
        json=body,
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=600,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if not payload or payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            if etype == "STEP_STARTED":
                print(f"\n▶ {ev.get('stepName')}")
            elif etype == "TEXT_MESSAGE_CONTENT":
                delta = ev.get("delta", "")
                print(f"  {delta[:200]}")
            elif etype == "TOOL_CALL_START":
                name = ev.get("toolCallName")
                print(f"  🔧 {name}")
            elif etype == "TOOL_CALL_RESULT":
                content = ev.get("content", "")[:200]
                print(f"  📥 {content}...")
            elif etype == "RUN_FINISHED":
                print("\n✅ done.")
            elif etype == "RUN_ERROR":
                print(f"\n⚠ {ev.get('message')}")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Is metformin contraindicated in CKD with eGFR 35?"
    run_question(q)
