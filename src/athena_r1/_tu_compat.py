"""Compatibility shim for ToolUniverse's HTTP client.

Recent ``tooluniverse`` releases (>=1.3) ship a ``ToolUniverseClient`` whose
dynamic proxy forwards every method call to the server as *keyword-only*
(``method_proxy(**kwargs)``). ATHENA's engine, however — like the in-process
``ToolUniverse`` API the client is meant to mirror — calls several methods
positionally, e.g. ``tool_specification("CallAgent", return_prompt=True)`` and
``run_one_function(tool_call)``. Against the keyword-only proxy those raise::

    TypeError: ToolUniverseClient.__getattr__.<locals>.method_proxy()
               takes 0 positional arguments but 1 was given

``CompatToolUniverseClient`` restores positional support by mapping positional
arguments onto the target method's declared parameter names. The names are read
from the server's own ``/api/methods`` introspection (already fetched and cached
by the base client), so nothing here is hard-coded — the shim keeps working if a
method's signature changes upstream.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tooluniverse import ToolUniverseClient


class CompatToolUniverseClient(ToolUniverseClient):
    """``ToolUniverseClient`` that also accepts positional arguments."""

    def _param_names(self, method_name: str) -> list[str]:
        """Ordered parameter names the server advertises for ``method_name``."""
        for info in self._get_available_methods():
            if info.get("name") == method_name:
                return [p.get("name") for p in info.get("parameters", [])]
        return []

    def __getattr__(self, method_name: str) -> Callable[..., Any]:
        # Base class raises AttributeError for private names; preserve that.
        kw_proxy = super().__getattr__(method_name)
        if method_name.startswith("_"):
            return kw_proxy

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
            return kw_proxy(**kwargs)

        return proxy
