"""Print the current ATHENA-R1 agent configuration as JSON.

Useful for monitoring, audit trails, and "did I configure this right?" sanity
checks. Reads from the running ATHENA-R1 agent (loads via the same env vars
the other servers use) without forcing init() — so it's cheap.

Usage:
    python examples/info_cli.py           # human-readable
    python examples/info_cli.py --json    # raw JSON

Or query a running AG-UI / OpenAI server directly:
    curl -s http://localhost:8090/health | jq
    curl -s http://localhost:9000/health | jq
"""

import argparse
import json
import os
import sys

from athena_r1 import AthenaR1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true", help="raw JSON (no formatting)")
    p.add_argument(
        "--init", action="store_true", help="also actually load tools + model (slow, ~30s)"
    )
    args = p.parse_args()

    agent = AthenaR1(
        model=os.environ.get("ATHENA_MODEL_PATH", "mims-harvard/ATHENA-R1-Qwen3-8B"),
        vllm_url=os.environ.get("VLLM_URL", "http://0.0.0.0:8000/v1"),
        tool_server=os.environ.get("TOOLUNIVERSE_API", "http://0.0.0.0:8080"),
        max_agent_level=int(os.environ.get("ATHENA_MAX_AGENT_LEVEL", "0")),
    )
    if args.init:
        agent.init()

    info = agent.info()
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"{'Field':22s}  Value")
        print("-" * 60)
        for k, v in info.items():
            display = v
            if isinstance(v, list):
                if len(v) > 4:
                    display = f"{v[:3]} ... (+{len(v) - 3} more)"
            print(f"{k:22s}  {display}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
