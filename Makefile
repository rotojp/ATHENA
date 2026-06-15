# Convenience commands for ATHENA-R1 development.
#
# Common entry points:
#   make install            install the package with web + dev extras
#   make lint               ruff check
#   make fmt                ruff format
#   make test-offline       run only the tests that don't need vLLM/TU
#   make test-full          run the whole pytest suite (needs live servers)
#   make smoke              `athena-r1 info` against the current env vars
#   make serve-agui         launch the AG-UI web server on :8090
#   make serve-openai       launch the OpenAI-compat server on :9000
#   make repro              full 456-Q reproducibility run (resumable)
#
# All commands run from the repo root.

PYTHON ?= python
PYTHONPATH := src

export PYTHONPATH

.PHONY: install lint fmt fmt-check test-offline test-full smoke serve-agui serve-openai repro clean

install:
	$(PYTHON) -m pip install -e ".[web,dev]"

lint:
	$(PYTHON) -m ruff check .

fmt:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

fmt-check:
	$(PYTHON) -m ruff format --check .

test-offline:
	$(PYTHON) -m pytest \
	  tests/test_smoke.py::test_imports_only \
	  tests/test_cli.py \
	  tests/test_info.py \
	  tests/test_validation.py \
	  tests/test_lifecycle.py \
	  tests/test_answer_result.py \
	  tests/test_api_edges.py \
	  tests/test_agui_multiturn.py \
	  tests/test_agui_id_encoding.py \
	  tests/test_agui_upstream_failures.py \
	  tests/test_agui_resource_leak.py \
	  tests/test_agui_edge_events.py \
	  tests/test_agui_concurrency.py \
	  tests/test_openai_server_error_path.py \
	  tests/test_timeout_and_multiturn.py::test_openai_server_multiturn_history_folded \
	  tests/test_timeout_and_multiturn.py::test_openai_server_singleturn_skips_history \
	  tests/test_timeout_and_multiturn.py::test_openai_server_system_message_preserved \
	  -v

test-full:
	$(PYTHON) -m pytest tests/ -v

smoke:
	$(PYTHON) -m athena_r1 info

serve-agui:
	$(PYTHON) web/agui_server.py

serve-openai:
	$(PYTHON) web/openai_server.py

repro:
	$(PYTHON) -u tests/test_reproducibility_full.py

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
