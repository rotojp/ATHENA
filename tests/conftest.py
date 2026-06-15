"""Shared pytest fixtures for ATHENA-R1 tests.

The integration tests need:
    * a vLLM server at $VLLM_URL (default http://0.0.0.0:8000/v1)
    * a ToolUniverse server at $TOOLUNIVERSE_API (default http://0.0.0.0:8080)

If either is missing, integration tests are skipped automatically.
"""

import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


VLLM_URL = os.environ.get("VLLM_URL", "http://0.0.0.0:8000/v1")
TOOL_SERVER = os.environ.get("TOOLUNIVERSE_API", "http://0.0.0.0:8080")


def _is_url_alive(url: str, path: str = "") -> bool:
    try:
        r = requests.get(url.rstrip("/") + path, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def vllm_url() -> str:
    if not _is_url_alive(VLLM_URL.rstrip("/v1") + "/v1/models"):
        pytest.skip(f"vLLM server not reachable at {VLLM_URL}")
    return VLLM_URL


@pytest.fixture(scope="session")
def tool_server() -> str:
    if not _is_url_alive(TOOL_SERVER, "/health"):
        pytest.skip(f"ToolUniverse server not reachable at {TOOL_SERVER}")
    return TOOL_SERVER


@pytest.fixture(scope="session")
def model_id(vllm_url: str) -> str:
    r = requests.get(f"{vllm_url.rstrip('/v1')}/v1/models", timeout=5)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


@pytest.fixture(scope="session")
def agent(model_id: str, vllm_url: str, tool_server: str):
    """Construct and initialize an AthenaR1 instance once per test session."""
    from athena_r1 import AthenaR1

    os.environ.setdefault("AZURE_API_KEY", "dummy")
    os.environ.setdefault("ATHENA_R1_LOG_LEVEL", "WARNING")

    a = AthenaR1(
        model=model_id,
        vllm_url=vllm_url,
        tool_server=tool_server,
        max_round=10,
    )
    a.init()
    return a
