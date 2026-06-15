# ATHENA-R1

**An AI agent for treatment reasoning over a biomedical tool universe**

[![Project Page](https://img.shields.io/badge/Project_Page-athena.openscientist.ai-2ea44f)](https://athena.openscientist.ai/)
[![Paper](https://img.shields.io/badge/Arxiv-ATHENA--R1-red)](#)
[![Code](https://img.shields.io/badge/Code-ATHENA--R1-purple)](https://github.com/mims-harvard/ATHENA)
[![Model](https://img.shields.io/badge/HuggingFace-ATHENA--R1-yellow)](https://huggingface.co/mims-harvard/ATHENA-R1-Qwen3-8B)
[![ToolUniverse](https://img.shields.io/badge/ToolUniverse-mims--harvard-blue)](https://github.com/mims-harvard/ToolUniverse)

ATHENA-R1 is an AI agent for treatment reasoning, trained through reinforcement
learning over a universe of biomedical tools. Rather than answering in a single
step, it performs multi-step reasoning — identifying what evidence is needed,
selecting from a library of 212 biomedical tools (served via
[ToolUniverse](https://github.com/mims-harvard/ToolUniverse)), retrieving
evidence from curated biomedical resources, and incorporating that evidence into
subsequent reasoning steps to reach evidence-grounded clinical decisions.

---

## Installation

```bash
pip install "git+https://github.com/mims-harvard/ATHENA.git"
```

For local model serving:

```bash
pip install "athena-r1[vllm,web] @ git+https://github.com/mims-harvard/ATHENA.git"
```

## Quick start

ATHENA-R1 needs two backing services: a vLLM server hosting the model and a
ToolUniverse HTTP server hosting the tools.

```bash
# One-shot launcher (TU + vLLM + AG-UI + OpenAI-compat server)
bash scripts/launch_all.sh mims-harvard/ATHENA-R1-Qwen3-8B

# Or start them individually:
bash scripts/launch_tooluniverse.sh 8080
bash scripts/launch_vllm.sh 8000 mims-harvard/ATHENA-R1-Qwen3-8B
python examples/quickstart.py
```

### Single question, no options

```python
from athena_r1 import AthenaR1

# `with` block guarantees clean shutdown of the underlying clients.
with AthenaR1(
    model="mims-harvard/ATHENA-R1-Qwen3-8B",
    vllm_url="http://0.0.0.0:8000/v1",
    tool_server="http://0.0.0.0:8080",
) as agent:
    result = agent.answer(
        "A 65-year-old with eGFR 35 and T2DM starting metformin: what dose adjustment?",
        timeout=180,           # wall-clock budget; force-finish if exceeded
    )

print(result.answer)
print(f"  rounds_used = {result.rounds_used}")
print(f"  elapsed_s   = {result.elapsed_s}")
print(f"  tools_used  = {result.tools_used}")
```

### Streaming (live reasoning feedback)

For interactive UIs, use `answer_streaming()` to receive a sequence of
`RoundEvent` objects as the agent thinks:

```python
from athena_r1 import AthenaR1

agent = AthenaR1(model="...", vllm_url="...", tool_server="...")

for ev in agent.answer_streaming("Dose adjustment for metformin in CKD?"):
    if ev.type == "round_start":
        print(f"\n--- Round {ev.round} ---")
    elif ev.type == "reasoning":
        print(ev.content)
    elif ev.type == "tool_call":
        print(f"🔧 calling: {ev.content}")
    elif ev.type == "tool_result":
        print(f"📥 {ev.metadata['name']}: {ev.content[:200]}...")
    elif ev.type == "final_answer":
        print(f"\n✅ Final: {ev.content}")
```

Same backing API and accuracy as `answer()` — just yields events live
instead of blocking until the full answer is ready.

### Multiple-choice evaluation

For MCQ benchmarks where you need a single letter back, run `answer()` and then
map the resulting conversation to a letter using either the local model or
GPT-5:

```python
from athena_r1 import AthenaR1, Backend

agent = AthenaR1(model="...", vllm_url="...", tool_server="...")

result = agent.answer(question)                                     # Stage 1
letter = agent.map_to_option(                                       # Stage 2
    result.conversation,
    options={"A": "...", "B": "...", "C": "...", "D": "..."},
    backend=Backend.ATHENA,                                         # or Backend.GPT
)
```

This two-stage design — free-form reasoning followed by a *separate* option
mapping — matches the paper's evaluation protocol and avoids contaminating
the reasoning trace with MCQ-specific prompts.

### Detailed reports

`answer()` returns a concise free-form answer. To turn a completed run into a
structured, citation-grounded **clinical report** — *Recommendation*, *Key
Evidence* (with the tool sources each finding drew on), *Reasoning*, and
*Caveats* — pass the result to `generate_report()`:

```python
from athena_r1 import AthenaR1

agent = AthenaR1(model="...", vllm_url="...", tool_server="...")

result = agent.answer("Is metformin contraindicated at eGFR 25?")

report = agent.generate_report(result)          # full markdown report (str)
print(report)

# …or stream it for a live UI:
for chunk in agent.generate_report(result, stream=True):
    print(chunk, end="", flush=True)
```

The report is synthesised **only from that run's trace**: citations are
restricted to the tools the agent actually called, and source labels are
validated so the model cannot fabricate a reference. The same capability is
available over HTTP — see [Web interfaces](#web-interfaces) below.

## Web interfaces

ATHENA-R1 ships two ready-to-deploy web servers and a Docker Compose stack:

| | What it gives you | Code to write |
|---|---|---|
| `web/agui_server.py` | AG-UI protocol server + bundled chat demo with subagent visualisation | 0 |
| `web/openai_server.py` | OpenAI-compatible HTTP API (multi-turn, streaming) | 0 |
| `web/docker-compose.yml` | OpenAI server + Open WebUI (full chat UI) | 0 |

Both servers also expose the [detailed report](#detailed-reports): the AG-UI
server adds `POST /report` (send a captured run trace, get the report streamed
back), and the OpenAI server accepts a `report: true` flag on
`/v1/chat/completions` to return the structured report instead of the short
answer. The bundled demo surfaces it as a **Detailed report** button on each
answer.

See [`web/README.md`](web/README.md) for setup instructions and the full set
of UI client options (CopilotKit, assistant-ui, agent-chat-ui, LangGraph Studio).

## API reference

### `AthenaR1` (the agent)

```python
agent = AthenaR1(
    model="mims-harvard/ATHENA-R1-Qwen3-8B",
    vllm_url="http://0.0.0.0:8000/v1",
    tool_server="http://0.0.0.0:8080",
    max_agent_level=0,         # set >0 to enable CallAgent recursion
    presence_penalty=0.0,      # paper canonical
    cache_tool=False,
)
```

| Method | Purpose |
|---|---|
| `agent.answer(q, timeout=120)` | Stage-1 multi-step reasoning, returns `AnswerResult` |
| `agent.answer_streaming(q)` | Same as `answer()` but yields `RoundEvent`s live |
| `agent.map_to_option(conv, opts, backend=...)` | Stage-2 MCQ letter extraction |
| `agent.generate_report(result, stream=False)` | Structured clinical report (Recommendation / Key Evidence / Reasoning / Caveats) synthesised from a run's trace |
| `agent.init()` / `agent.close()` | Explicit lifecycle (or use `with AthenaR1(...) as agent:`) |
| `agent.info()` | JSON-serialisable dict of the current config (handy for `/health`) |

### Command-line

After `pip install`, an `athena-r1` console script is registered:

```bash
athena-r1                                       # show version + summary
athena-r1 version                               # print version and exit
athena-r1 info                                  # JSON-dump the current agent config
athena-r1 ask "..." --timeout 120               # one-shot question (needs vLLM + TU)
athena-r1 ask "..." --temperature 0.3 --max-round 10
```

### `AnswerResult` (returned by `answer()`)

| field | description |
|---|---|
| `answer` | the model's free-form final answer (Stage-1 output) |
| `conversation` | full multi-turn chat history including tool calls and results — pass to `map_to_option()` for an MCQ letter |
| `rounds_used` | number of reasoning rounds consumed (1..max_round) |
| `elapsed_s` | wall-clock seconds spent inside `answer()` |
| `tools_used` | distinct biomedical tools called during Stage-1 (meta-tools `Tool_RAG`/`CallAgent`/`Finish` excluded) |
| `cancelled` | `True` iff a `timeout=` argument fired and the engine force-finished from a partial trace |
| `forced` | `True` iff the engine synthesized the answer after hitting `max_round` without a natural `[FinalAnswer]` (orthogonal to `cancelled`) |

### `RoundEvent` (for streaming UIs)

| field | description |
|---|---|
| `type` | one of `round_start`, `reasoning`, `tool_call`, `tool_rag_query`, `tools_retrieved`, `tool_result`, `subagent_start`, `subagent_end`, `final_answer` |
| `content` | event-type-dependent payload string |
| `round` | reasoning round number (`1`-indexed) |
| `agent_id` | `"main"`, `"main.sub-1"`, `"main.sub-1.sub-1"`, … |
| `agent_level` | 0 for main agent, +1 per nesting |
| `parent_agent_id` | for sub-agents, the spawner's `agent_id` |
| `metadata` | extra fields (`tool_name`, `tools` list, …) |

### `Backend` enum

`Backend.ATHENA` (default; the local RL-trained model served by vLLM) or
`Backend.GPT` (Azure GPT-5, needs `AZURE_API_KEY`). The two stages can use
different backends — `map_to_option` accepts a different backend than the one
passed to `answer`.

## Evaluation results

Headline results from the paper. ATHENA-R1 is evaluated in an open-ended setting
(each question answered free-form, then mapped to one of the original answer
choices) against GPT-5, DeepSeek-R1 (671B) and Qwen3.

| Benchmark | n | ATHENA-R1 | GPT-5 | DeepSeek-R1 | Qwen3 |
|---|---|---|---|---|---|
| **DrugPC** (open-ended drug reasoning) | 3,168 | **94.7%** | 76.9% | 68.8% | 48.7% |
| **TreatmentPC** (patient-specific treatment) | 456 | **82.9%** | 72.2% | 67.5% | 39.2% |

ATHENA-R1 exceeds GPT-5 by **+17.8** points on DrugPC and **+10.7** on
TreatmentPC. It also generalises across brand names, generic names and free-text
drug descriptions (BrandPC / GenericPC / DescriptionPC benchmarks).

See [`docs/eval_results.md`](docs/eval_results.md) for the full benchmark
tables, the two-level self-learning ablation, and reproduction details.

## Intended use

ATHENA-R1 is a research artifact for treatment-reasoning research and
decision support. It is not a medical device and must not be used for direct
patient care.

## Citation

```bibtex
@article{gao2026athena,
  title   = {An AI agent for treatment reasoning over a biomedical tool universe},
  author  = {Gao, Shanghua and ... and Zitnik, Marinka},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

ATHENA-R1 retrieves evidence through
[ToolUniverse](https://github.com/mims-harvard/ToolUniverse), a library of
curated biomedical tools.
