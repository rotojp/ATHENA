"""Streaming demo: watch ATHENA-R1 reason live.

Each reasoning round, tool call, and tool result lands in the terminal as
it happens — useful for interactive clients and paper figures.

Prerequisites (same as quickstart):
    bash scripts/launch_tooluniverse.sh
    bash scripts/launch_vllm.sh 8000 mims-harvard/ATHENA-R1-Qwen3-8B
"""

import os
import time

from athena_r1 import AthenaR1


def main() -> None:
    agent = AthenaR1(
        model=os.environ.get("ATHENA_MODEL_PATH", "mims-harvard/ATHENA-R1-Qwen3-8B"),
        vllm_url=os.environ.get("VLLM_URL", "http://0.0.0.0:8000/v1"),
        tool_server=os.environ.get("TOOLUNIVERSE_API", "http://0.0.0.0:8080"),
    )

    question = (
        "A 65-year-old patient with type 2 diabetes and eGFR of 35 mL/min "
        "is starting metformin. What dose adjustment is needed?"
    )
    print(f"Q: {question}\n" + "=" * 60)

    t0 = time.time()
    for ev in agent.answer_streaming(question, temperature=0.7):
        dt = time.time() - t0
        if ev.type == "round_start":
            print(f"\n[{dt:5.1f}s] === Round {ev.round} ===")
        elif ev.type == "reasoning":
            text = (ev.content or "").strip()
            if text:
                print(f"[{dt:5.1f}s] reasoning: {text[:200]}...")
        elif ev.type == "tool_call":
            print(f"[{dt:5.1f}s] 🔧 tool_call: {(ev.content or '')[:200]}")
        elif ev.type == "tool_result":
            name = ev.metadata.get("name", "tool")
            print(f"[{dt:5.1f}s] 📥 {name}: {(ev.content or '')[:200]}...")
        elif ev.type == "final_answer":
            text = ev.content or ""
            if "</think>" in text:
                text = text.split("</think>", 1)[-1].strip()
            print(f"\n[{dt:5.1f}s] ✅ FINAL ANSWER:\n{text}")


if __name__ == "__main__":
    main()
