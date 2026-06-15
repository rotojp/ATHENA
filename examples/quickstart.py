"""Quickstart: ask ATHENA-R1 a single clinical question (no options).

Prerequisites:
    1. ToolUniverse HTTP server running:
        python -m tooluniverse.http_api_server_cli --port 8080
    2. vLLM serving the ATHENA-R1 model:
        vllm serve mims-harvard/ATHENA-R1-Qwen3-8B \
            --port 8000 --gpu-memory-utilization 0.85

Environment overrides (useful when testing against an alternate
checkpoint or a different host):

    ATHENA_MODEL_PATH    model id passed to vLLM (default: HF id above)
    VLLM_URL             vLLM endpoint (default: http://0.0.0.0:8000/v1)
    TOOLUNIVERSE_API     ToolUniverse endpoint (default: http://0.0.0.0:8080)
"""

import os

from athena_r1 import AthenaR1


def main() -> None:
    agent = AthenaR1(
        model=os.environ.get("ATHENA_MODEL_PATH", "mims-harvard/ATHENA-R1-Qwen3-8B"),
        vllm_url=os.environ.get("VLLM_URL", "http://0.0.0.0:8000/v1"),
        tool_server=os.environ.get("TOOLUNIVERSE_API", "http://0.0.0.0:8080"),
    )

    question = (
        "A 65-year-old patient with chronic kidney disease (eGFR 35) and "
        "type-2 diabetes is starting metformin. What dose adjustment is "
        "needed and what monitoring should be in place?"
    )

    result = agent.answer(question, temperature=0.7)

    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60)
    print(f"\nReasoning rounds used: {result.rounds_used}")
    print(f"\nFinal answer:\n{result.answer}")


if __name__ == "__main__":
    main()
