# Web deployment for ATHENA-R1

ATHENA-R1 ships two HTTP backends. Both stream the agent's reasoning live;
they differ only in protocol.

## Option A — AG-UI server

Serves the agent over the [AG-UI](https://docs.ag-ui.com/) protocol. Streams
nested sub-agent reasoning and tool-call results, and works with any AG-UI
frontend (see the table below), so you can swap UIs without changing the
server.

```bash
pip install ".[web]"
export VLLM_URL=http://0.0.0.0:8000/v1
export TOOLUNIVERSE_API=http://0.0.0.0:8080
python web/agui_server.py
# → POST  http://0.0.0.0:8090/    (AG-UI endpoint)
# → GET   http://0.0.0.0:8090/health
```

Pair with any AG-UI client:

| Frontend | Setup | Best for |
|---|---|---|
| [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui) | clone, point base URL at our `:8090` | minimal React chat, paper demo |
| [CopilotKit](https://github.com/CopilotKit/CopilotKit) | `npm i @copilotkit/react-core` | embed-in-app copilot, generative UI |
| [assistant-ui](https://github.com/assistant-ui/assistant-ui) | `npm i @assistant-ui/react` | shadcn-style custom chat |
| [LangGraph Studio](https://langchain-ai.github.io/langgraph/) | configure custom URL | graph-view debugging |

Event types emitted (matching the AG-UI spec): `RUN_STARTED`,
`STEP_STARTED`/`STEP_FINISHED` (one pair per reasoning round and per
sub-agent), `TEXT_MESSAGE_START`/`CONTENT`/`END`, `TOOL_CALL_START`/`ARGS`/
`END`/`RESULT`, `RUN_FINISHED`, `RUN_ERROR`.

## Option B — OpenAI-compatible API server

Exposes the agent behind an [OpenAI ChatCompletion](https://platform.openai.com/docs/api-reference/chat)
endpoint, so any client that speaks the OpenAI SDK (Open WebUI, LibreChat,
the `openai` Python / JS clients, LangChain's `ChatOpenAI`, etc.) can call it.

```bash
pip install ".[web]"
export VLLM_URL=http://0.0.0.0:8000/v1
export TOOLUNIVERSE_API=http://0.0.0.0:8080
python web/openai_server.py
# → POST http://0.0.0.0:9000/v1/chat/completions
```

Client (any OpenAI SDK):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9000/v1", api_key="any")
stream = client.chat.completions.create(
    model="athena-r1",
    messages=[{"role": "user", "content": "Metformin in CKD eGFR 35?"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Required upstream services

Both options need:

1. vLLM serving the model on port 8000: `bash scripts/launch_vllm.sh`
2. ToolUniverse server on port 8080: `bash scripts/launch_tooluniverse.sh`
