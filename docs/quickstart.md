# Quickstart

## 1. Install

```bash
pip install "git+https://github.com/mims-harvard/ATHENA.git"
# or for local model serving + web UIs:
pip install "athena-r1[vllm,web] @ git+https://github.com/mims-harvard/ATHENA.git"
```

## 2. Launch backing services

ATHENA-R1 needs two HTTP services running. The Python agent itself is a thin
client.

```bash
# ToolUniverse server (port 8080)
bash scripts/launch_tooluniverse.sh

# vLLM serving the ATHENA-R1 model (port 8000)
bash scripts/launch_vllm.sh 8000 mims-harvard/ATHENA-R1-Qwen3-8B
```

These scripts block until each service is healthy. The vLLM warm-up takes
a few minutes the first time the model is loaded.

## 3. Ask a question

```python
from athena_r1 import AthenaR1

agent = AthenaR1(
    model="mims-harvard/ATHENA-R1-Qwen3-8B",
    vllm_url="http://0.0.0.0:8000/v1",
    tool_server="http://0.0.0.0:8080",
)

print(agent.answer("Dose adjustment for metformin in CKD eGFR 35?").answer)
```

## 4. (Optional) Web chat UI

```bash
python web/agui_server.py    # AG-UI server + bundled demo → http://localhost:8090
```

or for a full ChatGPT-style UI with multi-user support (Open WebUI):

```bash
cd web && docker compose up -d   # → http://localhost:3000
```

See [`web/README.md`](../web/README.md) for the full set of
options (CopilotKit, assistant-ui, agent-chat-ui, LangGraph Studio).

## 5. (Optional) Run the eval benchmarks

```bash
python examples/eval_mcq.py
```

For bulk evaluation of an MCQ dataset, write a short loop:

```python
from athena_r1 import AthenaR1, Backend

agent = AthenaR1(model="...", vllm_url="...", tool_server="...")

for question, options, gold in dataset:
    result = agent.answer(question)
    pred = agent.map_to_option(result.conversation, options, backend=Backend.ATHENA)
    print(question, pred, gold, pred == gold)
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `connection refused` on :8000 | vLLM not started | `bash scripts/launch_vllm.sh` |
| `connection refused` on :8080 | ToolUniverse not started | `bash scripts/launch_tooluniverse.sh` |
| `meta tensor` error from vLLM | torch version mismatch | install `vllm>=0.6` |
| Stage-1 hits `max_round=40` often | model running out of context | reduce `max_round` or increase `max_new_tokens` per call |
