"""Tests for the ``python -m athena_r1`` CLI entry point.

These cover the cheap paths — summary (no args), --help, and unknown-command
error reporting. The ``ask`` subcommand needs a live vLLM + ToolUniverse
server, so it's covered indirectly by the eval/integration tests.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    """Invoke ``athena_r1.__main__.main`` with captured stdout/stderr."""
    sys.path.insert(0, "src")
    from athena_r1.__main__ import main

    out_buf, err_buf = io.StringIO(), io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        rc = main(argv)
    return rc, out_buf.getvalue(), err_buf.getvalue()


def test_cli_no_args_prints_summary():
    rc, out, _ = _run_cli([])
    assert rc == 0
    assert "athena_r1 v" in out
    assert "Commands:" in out
    assert "info" in out
    assert "ask QUESTION" in out


def test_cli_help_flag_prints_summary():
    rc, out, _ = _run_cli(["--help"])
    assert rc == 0
    assert "athena_r1 v" in out
    assert "Environment overrides:" in out


def test_cli_unknown_command_exits_nonzero():
    rc, _, err = _run_cli(["bogus"])
    assert rc == 2
    assert "unknown command" in err


def test_cli_ask_without_question_errors():
    rc, _, err = _run_cli(["ask"])
    assert rc == 2
    assert "requires a question" in err


def test_cli_summary_mentions_env_vars():
    """Help text should list every env var the CLI honours."""
    _, out, _ = _run_cli([])
    for var in ("ATHENA_MODEL_PATH", "VLLM_URL", "TOOLUNIVERSE_API", "AZURE_API_KEY"):
        assert var in out


def test_cli_summary_documents_ask_flags():
    """Help text should describe every flag accepted by `ask`."""
    _, out, _ = _run_cli([])
    for flag in ("--timeout", "--temperature", "--max-round"):
        assert flag in out, f"{flag!r} missing from help: {out!r}"


def test_cli_version_prints_version_only():
    """`athena-r1 version` should print the version string and nothing else."""
    sys.path.insert(0, "src")
    from athena_r1._version import __version__

    rc, out, _ = _run_cli(["version"])
    assert rc == 0
    assert out.strip() == __version__


def test_cli_version_flag_short_and_long():
    """`--version` and `-V` flags should also print just the version."""
    sys.path.insert(0, "src")
    from athena_r1._version import __version__

    for flag in ("--version", "-V"):
        rc, out, _ = _run_cli([flag])
        assert rc == 0
        assert out.strip() == __version__, f"flag {flag!r} gave {out!r}"


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
