"""``python -m athena_r1`` (or the ``athena-r1`` console script).

Quick CLI for sanity-checking a deployment without writing Python:

    athena-r1                       # show version + summary
    athena-r1 version               # print version and exit
    athena-r1 info                  # JSON-dump current agent config
    athena-r1 ask "..."             # one-shot question (needs vLLM + TU)

Or via the module form:

    python -m athena_r1 [...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from ._version import __version__


def _make_agent():
    # Lazy import — keeps ``athena-r1 --help`` snappy by deferring the heavy
    # transformers / tooluniverse import chain until we actually need the agent.
    from .agent import AthenaR1

    return AthenaR1(
        model=os.environ.get("ATHENA_MODEL_PATH", "mims-harvard/ATHENA-R1-Qwen3-8B"),
        vllm_url=os.environ.get("VLLM_URL", "http://0.0.0.0:8000/v1"),
        tool_server=os.environ.get("TOOLUNIVERSE_API", "http://0.0.0.0:8080"),
        max_agent_level=int(os.environ.get("ATHENA_MAX_AGENT_LEVEL", "0")),
    )


def cmd_summary() -> int:
    print(f"athena_r1 v{__version__}")
    print()
    print("Usage:  athena-r1 <command>     (or:  python -m athena_r1 <command>)")
    print()
    print("Commands:")
    print("  version          print version string and exit")
    print("  info             show current agent configuration as JSON")
    print("  ask QUESTION     ask the agent one question (needs vLLM + TU live)")
    print()
    print("Flags:")
    print("  --timeout SEC          wall-clock cap for `ask`")
    print("  --temperature FLOAT    sampling temperature for `ask` (default 0.7)")
    print("  --max-round INT        override agent's max_round for `ask`")
    print("  -V, --version          print version string and exit")
    print()
    print("Environment overrides:")
    print("  ATHENA_MODEL_PATH    HF id / local path of the model")
    print("  VLLM_URL             vLLM endpoint (default http://0.0.0.0:8000/v1)")
    print("  TOOLUNIVERSE_API     ToolUniverse endpoint (default http://0.0.0.0:8080)")
    print("  AZURE_API_KEY        only needed for Backend.GPT")
    return 0


def cmd_info() -> int:
    agent = _make_agent()
    print(json.dumps(agent.info(), indent=2))
    return 0


def cmd_ask(
    question: str,
    timeout: float | None,
    temperature: float,
    max_round: int | None,
) -> int:
    agent = _make_agent()
    kwargs: dict[str, Any] = {"temperature": temperature, "timeout": timeout}
    if max_round is not None:
        kwargs["max_round"] = max_round
    result = agent.answer(question, **kwargs)
    print(result.answer)
    print()
    print(f"  rounds_used = {result.rounds_used}")
    print(f"  elapsed_s   = {result.elapsed_s}")
    print(f"  tools_used  = {result.tools_used}")
    print(f"  cancelled   = {result.cancelled}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m athena_r1", add_help=False)
    parser.add_argument("command", nargs="?", default=None)
    parser.add_argument("question", nargs="*", default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-round", type=int, default=None)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("-V", "--version", action="store_true")
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0
    if args.help or args.command is None:
        return cmd_summary()
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "info":
        return cmd_info()
    if args.command == "ask":
        if not args.question:
            print("error: `ask` requires a question argument", file=sys.stderr)
            return 2
        return cmd_ask(
            " ".join(args.question),
            timeout=args.timeout,
            temperature=args.temperature,
            max_round=args.max_round,
        )
    print(f"unknown command: {args.command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
