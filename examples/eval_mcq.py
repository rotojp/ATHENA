"""Evaluation example: answer an MCQ and map to a letter (A/B/C/D).

Demonstrates the two-stage pattern used in the paper's benchmarks:
    1. `answer()` produces free-form reasoning.
    2. `map_to_option()` extracts a single letter, either via the local model
       (Backend.ATHENA, the model's own self-extraction) or via Azure GPT-5
       (Backend.GPT, robust external reader).
"""

import os

from athena_r1 import AthenaR1, Backend


def main() -> None:
    agent = AthenaR1(
        model=os.environ.get("ATHENA_MODEL_PATH", "mims-harvard/ATHENA-R1-Qwen3-8B"),
        vllm_url=os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1"),
        tool_server=os.environ.get("TOOLUNIVERSE_API", "http://127.0.0.1:8080"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_api_key=os.environ.get("AZURE_API_KEY"),
    )

    question = (
        "A 65-year-old patient with chronic kidney disease (eGFR 35) and "
        "type-2 diabetes is starting metformin. What is the most appropriate "
        "starting dose?"
    )
    options = {
        "A": "500 mg twice daily",
        "B": "1000 mg twice daily",
        "C": "Contraindicated; avoid metformin",
        "D": "500 mg once daily, titrate based on GFR trajectory",
    }

    # Stage 1: free-form reasoning
    result = agent.answer(question, temperature=0.7)
    print(f"Reasoning rounds: {result.rounds_used}")
    print(f"Free-form answer:\n{result.answer}\n")

    # Stage 2: map to option letter (two backends)
    athena_letter = agent.map_to_option(
        result.conversation,
        options,
        backend=Backend.ATHENA,
    )
    print(f"ATHENA mapping:  {athena_letter}")

    gpt_letter = agent.map_to_option(
        result.conversation,
        options,
        backend=Backend.GPT,
        question=question,
    )
    print(f"GPT-5 mapping:   {gpt_letter}")


if __name__ == "__main__":
    main()
