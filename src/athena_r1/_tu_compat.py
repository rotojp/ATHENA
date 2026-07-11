"""Compatibility shim for ToolUniverse's HTTP client.

Recent ``tooluniverse`` releases (>=1.3) ship a ``ToolUniverseClient`` whose
dynamic proxy forwards every method call to the server as *keyword-only*
(``method_proxy(**kwargs)``) and with a hard-coded 30-second request timeout.
Two problems for ATHENA:

1. The engine — like the in-process ``ToolUniverse`` API the client mirrors —
   calls several methods positionally, e.g.
   ``tool_specification("CallAgent", return_prompt=True)`` and
   ``run_one_function(tool_call)``. Against the keyword-only proxy those raise
   ``TypeError: method_proxy() takes 0 positional arguments but 1 was given``.

2. Tool_RAG retrieval embeds the (often long) query through a 1.5B model. On a
   CPU-only host that can take longer than 30s, so the call is cut off before it
   returns.

``CompatToolUniverseClient`` re-implements the proxy to (a) map positional
arguments onto the target method's server-declared parameter names — read from
``/api/methods`` introspection, so nothing is hard-coded — and (b) use a
configurable request timeout (``ATHENA_TU_TIMEOUT`` env var, default 120s).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import requests
from tooluniverse import ToolUniverseClient

DEFAULT_TIMEOUT = float(os.environ.get("ATHENA_TU_TIMEOUT", "120"))


class CompatToolUniverseClient(ToolUniverseClient):
    """``ToolUniverseClient`` with positional-arg support and a longer timeout."""

    # Parameter names for the methods the engine calls positionally, used when
    # the server's /api/methods introspection is unavailable (e.g. a slow or
    # briefly unreachable server, whose empty result the base client caches).
    # Without this, a valid positional call would raise a misleading
    # "0 parameters" TypeError. Kept in sync with the ToolUniverse API.
    _FALLBACK_PARAMS = {
        "tool_specification": ["tool_name", "return_prompt", "format"],
        "run_one_function": [
            "function_call_json",
            "stream_callback",
            "use_cache",
            "validate",
        ],
        "prepare_tool_prompts": ["tool_list", "mode", "valid_keys"],
        "extract_function_call_json": ["lst", "return_message", "verbose", "format"],
    }

    def __init__(self, *args: Any, timeout: float = DEFAULT_TIMEOUT, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._call_timeout = timeout

    def _param_names(self, method_name: str) -> list[str]:
        """Ordered parameter names for ``method_name``.

        Prefers the server's ``/api/methods`` introspection; falls back to a
        static map so a slow or unreachable server can't turn a valid positional
        call into a spurious "0 parameters" error.
        """
        for info in self._get_available_methods():
            if info.get("name") == method_name:
                params = [p.get("name") for p in info.get("parameters", [])]
                if params:
                    return params
        return self._FALLBACK_PARAMS.get(method_name, [])

    def _call(self, method_name: str, kwargs: dict[str, Any]) -> Any:
        """POST a method call to the server with the configured timeout."""
        try:
            resp = self.session.post(
                f"{self.base_url}/api/call",
                json={"method": method_name, "kwargs": kwargs},
                timeout=self._call_timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            if not result.get("success", False):
                error = result.get("error", "Unknown error")
                error_type = result.get("error_type", "UnknownError")
                raise Exception(f"[{error_type}] {error}")
            return result.get("result")
        except requests.exceptions.ReadTimeout:
            return f"Error: Tool execution timed out after {self._call_timeout}s"
        except requests.exceptions.RequestException as e:
            return f"Error: HTTP request failed for '{method_name}': {e}"

    def __getattr__(self, method_name: str) -> Callable[..., Any]:
        # Base class raises AttributeError for private names; preserve that.
        if method_name.startswith("_"):
            return super().__getattr__(method_name)

        def proxy(*args: Any, **kwargs: Any) -> Any:
            if args:
                names = self._param_names(method_name)
                if len(args) > len(names):
                    raise TypeError(
                        f"{method_name}() got {len(args)} positional argument(s) "
                        f"but the server declares only {len(names)} parameter(s)"
                    )
                for name, value in zip(names, args, strict=False):
                    if name in kwargs:
                        raise TypeError(f"{method_name}() got multiple values for '{name}'")
                    kwargs[name] = value
            return self._call(method_name, kwargs)

        return proxy
