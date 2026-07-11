"""AG-UI protocol server for ATHENA-R1.

Translates the agent's internal ``RoundEvent`` stream into AG-UI standard
events so any AG-UI-compliant frontend (CopilotKit, assistant-ui,
agent-chat-ui, LangGraph Studio, …) can connect without code changes.

Run:
    python web/agui_server.py
    # → POST  http://0.0.0.0:8090/    (AG-UI endpoint)
    # → GET   http://0.0.0.0:8090/health

Spec: https://docs.ag-ui.com/concepts/events
"""

from __future__ import annotations

import asyncio
import json
import os

# Local import — the helper lives in a sibling file so it can be unit-tested
# without pulling in the optional AG-UI / FastAPI dependencies.
import sys as _sys
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ag_ui.core import (
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from athena_r1 import AthenaR1

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _messages import fold_messages_to_prompt as _fold_messages_to_prompt  # noqa: E402

# Backwards-compat alias for old callers / tests that imported the previous name.
_last_user_message = _fold_messages_to_prompt

_HERE = os.path.dirname(os.path.abspath(__file__))
_STATIC_DIR = os.path.join(_HERE, "static")


# ─── App + agent singleton ────────────────────────────────────────────────


app = FastAPI(title="ATHENA-R1 AG-UI server", version="1.0.0")
_agent: AthenaR1 | None = None
# Serializes lazy init so two concurrent requests don't both construct +
# load the model. Without this, the second caller can see a half-built
# `_agent` (constructed but pre-`init()`) and use it before model loading
# completes, or both callers race into `init_model` and double-load.
import threading as _threading  # noqa: E402

_agent_lock = _threading.Lock()


def get_agent() -> AthenaR1:
    """Lazy-init the ATHENA-R1 agent on first request."""
    global _agent
    # Fast path: already built. ``getattr(..., True)`` so unit-test stubs
    # without `_initialized` are treated as ready (tests inject a fake agent
    # directly into the module global).
    if _agent is not None and getattr(_agent, "_initialized", True):
        return _agent
    with _agent_lock:
        if _agent is None:
            _agent = AthenaR1(
                model=os.environ.get("ATHENA_MODEL_PATH", "mims-harvard/ATHENA-R1-Qwen3-8B"),
                vllm_url=os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1"),
                tool_server=os.environ.get("TOOLUNIVERSE_API", "http://127.0.0.1:8080"),
                # Default 1: the Multi-agent toggle has effect out-of-box
                # (main can dispatch sub-agents) WITHOUT the explosive
                # grandchild nesting that level=2 produces. Live testing
                # showed level=2 spawning ~25 agents (7 top-level subs ×
                # 4-8 grandchildren) for a single comparison question — a
                # >10-minute run that's a poor demo experience. level=1
                # keeps a clean, snappy main→subs tree. Set
                # ATHENA_MAX_AGENT_LEVEL=2 for deeper nesting, or =0 to
                # force single-agent regardless of the UI toggle.
                max_agent_level=int(os.environ.get("ATHENA_MAX_AGENT_LEVEL", "1")),
            )
        if not getattr(_agent, "_initialized", True):
            _agent.init()
    return _agent


@app.on_event("startup")
def _warm_agent_in_background() -> None:
    """Pre-load the agent (tokenizer + 1.5B ToolRAG embedding model) in a
    background thread at server startup.

    Without this, the FIRST user request pays the full ~30-60s init cost
    (model tokenizer + ToolRAG-T1-GTE-Qwen2-1.5B retrieval model) — a poor
    first impression for the demo. Warming runs in a daemon thread so:
      * ``/health`` and the static UI stay responsive immediately, and
      * a request arriving mid-warm simply blocks on the same
        ``_agent_lock`` ``get_agent()`` already uses, so there's no double
        init and no race.
    A warm failure (e.g. vLLM/TU not up yet) is swallowed here; the first
    real request will retry via ``get_agent()`` and surface any error then.
    """

    def _warm() -> None:
        try:
            get_agent()
        except Exception:  # noqa: BLE001 — best-effort warmup, never fatal
            pass

    _threading.Thread(target=_warm, name="agent-warmup", daemon=True).start()


# ─── Event translation: ATHENA RoundEvent → AG-UI events ──────────────────


# AG-UI's TextMessage / ToolCall events don't carry an explicit agent_id, but
# their ``message_id`` / ``tool_call_id`` are opaque strings — so we encode the
# emitting agent_id into the ID itself. Frontend clients (the bundled HTML
# demo, agent-chat-ui, etc.) can split on ``::`` to recover the lane without
# relying on the "attach to most-recent step" heuristic which breaks when
# concurrent sub-agents interleave their events.
_ID_SEP = "::"


def _msg_id(agent_id: str) -> str:
    return f"msg{_ID_SEP}{agent_id}{_ID_SEP}{uuid.uuid4().hex[:12]}"


def _tool_call_id(agent_id: str) -> str:
    return f"tc{_ID_SEP}{agent_id}{_ID_SEP}{uuid.uuid4().hex[:12]}"


def agent_id_from_event_id(event_id: str) -> str:
    """Recover the emitting ``agent_id`` from a message_id or tool_call_id.

    Returns ``"main"`` if the id wasn't produced by this server (e.g. a legacy
    msg-xxxxxxxxxxxx style ID from an older deployment).
    """
    if event_id is None or _ID_SEP not in event_id:
        return "main"
    parts = event_id.split(_ID_SEP)
    return parts[1] if len(parts) >= 3 else "main"


async def _translate_to_agui(
    question: str,
    thread_id: str,
    run_id: str,
    multi_agent_requested: bool = True,
    request: Any = None,
) -> AsyncIterator[Any]:
    """Run the agent in a worker thread; translate RoundEvent → AG-UI events.

    ``multi_agent_requested`` lets the caller force single-agent mode per
    request even when the server was booted with ``max_agent_level > 0``.
    Surfaced in the UI as the topbar "Multi-agent" toggle.

    ``request`` (the Starlette Request) is used to detect client disconnect
    PROACTIVELY: the SSE generator only observes a disconnect on its next
    ``yield``, but during a long generation no events are emitted, so it sits
    blocked on ``queue.get()`` and wouldn't set ``cancel_event`` until the
    generation finishes. A background watcher task polls
    ``request.is_disconnected()`` and trips ``cancel_event`` within ~1s, which
    the streaming inference loop checks every token — so an abandoned run
    stops in seconds instead of running the in-flight generation to
    completion.
    """

    yield RunStartedEvent(thread_id=thread_id, run_id=run_id)

    agent = get_agent()
    # Track the active text-message bracket per agent_id so we can wrap
    # streaming reasoning into TextMessageStart / Content / End triplets.
    open_message_id: dict[str, str] = {}
    # FIFO queues of pending tool_call_ids per (scope_label, channel). The
    # engine emits N `tool_call` events for a round, then N matching
    # `tool_result` events in execution order — we MUST queue ids, not store a
    # single slot (which would overwrite all but the last and leave earlier
    # tool chips spinning forever on the client).
    from collections import deque as _deque

    open_tool_call_id: dict[str, _deque] = {}

    def _push_tc(key: str, tc_id: str) -> None:
        open_tool_call_id.setdefault(key, _deque()).append(tc_id)

    def _pop_tc(key: str) -> str | None:
        q = open_tool_call_id.get(key)
        if not q:
            return None
        tc = q.popleft()
        if not q:
            open_tool_call_id.pop(key, None)
        return tc

    # Cap result content sent to the UI. Long tool results would balloon SSE
    # frames and the drawer's render budget. If we truncate, append an
    # explicit marker so the user knows there's more (the UI's "X KB of Y KB"
    # badge alone is brittle because it depends on guessing the original).
    _RESULT_CAP = 8000

    def _cap(s: str) -> str:
        s = s or ""
        if len(s) > _RESULT_CAP:
            return s[:_RESULT_CAP] + f"\n\n…[truncated · original {len(s):,} chars]"
        return s

    # Track the currently-open round step per agent_id so we can pair every
    # StepStartedEvent with a matching StepFinishedEvent (AG-UI spec).
    open_round_step: dict[str, str] = {}
    # Track the last reasoning text emitted per agent. We need this to decide
    # whether `final_answer`'s TextMessage would be a useful new chunk of
    # content or a duplicate of what already streamed. The engine has THREE
    # final_answer paths:
    #   1. Normal: model emits "<think>...</think>[FinalAnswer]X" inline; the
    #      reasoning event carries the whole thing and final_answer repeats
    #      "X". Re-emitting would overwrite the round's <think> block.
    #   2. Token-overflow fail: engine emits both `reasoning` and
    #      `final_answer` with the SAME error message. Same duplicate problem.
    #   3. Force-finish (max_round / cancellation): engine calls
    #      `get_answer_based_on_unfinished_reasoning` to synthesize a NEW
    #      answer the model never produced. The last reasoning is the model's
    #      incomplete trace, which does NOT contain the synthesized text.
    #      Without emitting the final_answer's TextMessage the UI's
    #      final-answer panel picks the incomplete reasoning instead.
    # Discriminate by containment: emit only when final_answer's content
    # isn't already inside the last reasoning text.
    last_reasoning_per_agent: dict[str, str] = {}
    # Track the last round number seen per agent. When the synthesized
    # final_answer needs its own step (force-finish path), we open a NEW
    # round (last+1) rather than letting the TextMessage land on — and
    # overwrite — the last reasoning round.
    last_round_num_per_agent: dict[str, int] = {}
    # Track every open ``/subagent`` step keyed by the SUB's agent_id so that
    # on RunError or engine crash (where subagent_end may never fire) the
    # cleanup at exit can still emit matching StepFinishedEvent and keep the
    # AG-UI step-event tree well-formed for strict clients.
    open_subagent_step: dict[str, str] = {}
    # Map sub agent_id → its CallAgent chip's tool_call_id. Set on
    # subagent_start (FIFO pop from parent's /callagent queue) and consumed
    # on subagent_end. Critical because: with parallel CallAgents, subs can
    # finish in a DIFFERENT order than they started — FIFO-popping the
    # /callagent queue on subagent_end would land sub-2's result content on
    # the chip the UI had already labelled "→ sub-1" (the label is fixed by
    # subagent_start arrival order, the result by subagent_end arrival order
    # — those orders disagree under out-of-order completion).
    callagent_chip_by_sub: dict[str, str] = {}

    # ``get_event_loop`` raises a DeprecationWarning when called from inside a
    # coroutine (Py 3.12+) and is slated to behave like
    # ``new_event_loop`` (Py 3.14+). Since we ARE inside a running loop —
    # this function is an async generator driven by FastAPI's
    # StreamingResponse — ``get_running_loop`` is the spec-correct call.
    loop = asyncio.get_running_loop()

    # Drive the synchronous answer_streaming() generator in a thread executor,
    # forwarding each emitted RoundEvent to this async pipeline via a queue.
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    # Enable CallAgent only if BOTH the server was booted with
    # max_agent_level > 0 AND the request opted in (UI multi-agent toggle).
    # The server config is the ceiling; the request can ask for less but
    # not more.
    server_allows_multi = getattr(agent._core, "max_agent_level", 0) > 0
    call_agent_enabled = server_allows_multi and multi_agent_requested

    # Shared cancel flag — set in the `finally` below when the client
    # aborts the SSE stream, so the engine's round-boundary poll bails
    # instead of running to completion in a zombie thread.
    import threading as _threading

    cancel_event = _threading.Event()

    def runner() -> None:
        try:
            for ev in agent.answer_streaming(
                question,
                temperature=0.7,
                call_agent=call_agent_enabled,
                external_cancel_event=cancel_event,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, ev)
        except Exception as e:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, ("__error__", e))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    loop.run_in_executor(None, runner)

    # Proactive disconnect watcher — see the docstring. Polls
    # ``request.is_disconnected()`` so ``cancel_event`` trips within ~1s of
    # the client leaving, even while the main loop below is blocked on
    # ``queue.get()`` through a long token generation.
    async def _disconnect_watcher() -> None:
        if request is None:
            return
        try:
            while not cancel_event.is_set():
                if await request.is_disconnected():
                    cancel_event.set()
                    return
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            # Swallow rather than re-raise: this task is fire-and-forget and
            # is cancelled (not awaited) from the generator's finally. Re-
            # raising would leave an unretrieved CancelledError that asyncio
            # logs as "Task ... was never retrieved" / "Task was destroyed".
            return
        except Exception:  # noqa: BLE001 — watcher must never crash the run
            pass

    watcher_task = loop.create_task(_disconnect_watcher())

    error_to_raise: Exception | None = None
    try:
        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            if isinstance(item, tuple) and item and item[0] == "__error__":
                error_to_raise = item[1]
                break
            ev = item

            scope_label = ev.agent_id  # "main", "main.sub-1", etc.

            if ev.type == "round_start":
                # Close out the previous round step for this agent_id, if any,
                # before opening the new one — keeps AG-UI's STEP_*  pairing intact.
                prev_step = open_round_step.pop(scope_label, None)
                if prev_step is not None:
                    yield StepFinishedEvent(step_name=prev_step)
                step_name = f"{scope_label}/round-{ev.round}"
                open_round_step[scope_label] = step_name
                if isinstance(ev.round, int):
                    last_round_num_per_agent[scope_label] = ev.round
                yield StepStartedEvent(step_name=step_name)

            elif ev.type == "reasoning":
                text = (ev.content or "").strip()
                if not text:
                    continue
                msg_id = open_message_id.get(scope_label)
                if msg_id is None:
                    msg_id = _msg_id(scope_label)
                    open_message_id[scope_label] = msg_id
                    yield TextMessageStartEvent(message_id=msg_id, role="assistant")
                yield TextMessageContentEvent(message_id=msg_id, delta=text)
                yield TextMessageEndEvent(message_id=msg_id)
                open_message_id.pop(scope_label, None)
                last_reasoning_per_agent[scope_label] = text

            elif ev.type == "tool_rag_query":
                tc_id = _tool_call_id(scope_label)
                yield ToolCallStartEvent(tool_call_id=tc_id, tool_call_name="Tool_RAG")
                yield ToolCallArgsEvent(
                    tool_call_id=tc_id, delta=json.dumps(ev.metadata.get("arguments") or {})
                )
                yield ToolCallEndEvent(tool_call_id=tc_id)
                _push_tc(f"{scope_label}/rag", tc_id)

            elif ev.type == "tools_retrieved":
                tc_id = _pop_tc(f"{scope_label}/rag")
                if tc_id:
                    msg_id = _msg_id(scope_label)
                    yield ToolCallResultEvent(
                        message_id=msg_id,
                        tool_call_id=tc_id,
                        content=_cap(json.dumps(ev.metadata.get("tools", []))),
                        role="tool",
                    )

            elif ev.type == "tool_call":
                tool_name = ev.metadata.get("tool_name") or "tool"
                # Skip Tool_RAG (own event path above). CallAgent is emitted
                # here too so the parent agent UI shows a clickable chip that
                # links to the spawned sub-agent.
                if tool_name == "Tool_RAG":
                    continue
                tc_id = _tool_call_id(scope_label)
                yield ToolCallStartEvent(tool_call_id=tc_id, tool_call_name=tool_name)
                try:
                    parsed = json.loads(ev.content or "{}")
                    args = parsed.get("arguments", {})
                except Exception:  # noqa: BLE001
                    args = {}
                yield ToolCallArgsEvent(tool_call_id=tc_id, delta=json.dumps(args))
                yield ToolCallEndEvent(tool_call_id=tc_id)
                # CallAgent goes into its own channel so subagent_end can
                # resolve it without colliding with regular tool results.
                channel = "callagent" if tool_name == "CallAgent" else "tool"
                _push_tc(f"{scope_label}/{channel}", tc_id)

            elif ev.type == "tool_result":
                # Prefer /tool. Fall back to /callagent (handles "CallAgent
                # disabled by max_agent_level": engine emits tool_result with
                # the disabled-error string but never fires subagent_end).
                # Final fallback /rag covers the Tool_RAG ERROR path: success
                # already drains /rag via tools_retrieved, but on error
                # tools_retrieved never fires and the chip would spin forever.
                tc_id = (
                    _pop_tc(f"{scope_label}/tool")
                    or _pop_tc(f"{scope_label}/callagent")
                    or _pop_tc(f"{scope_label}/rag")
                )
                if tc_id:
                    msg_id = _msg_id(scope_label)
                    yield ToolCallResultEvent(
                        message_id=msg_id,
                        tool_call_id=tc_id,
                        content=_cap(ev.content or ""),
                        role="tool",
                    )

            elif ev.type == "subagent_start":
                # Open a nested step. The child agent's own events (with
                # agent_id="main.sub-N") will land inside this bracket.
                sub_step = f"{ev.agent_id}/subagent"
                open_subagent_step[ev.agent_id] = sub_step
                # Bind THIS sub to the next CallAgent chip in the parent's
                # FIFO queue NOW (at start time) — matching the static UI's
                # link-by-arrival-order policy in TOOL_CALL_START handling.
                # Without binding here, subagent_end has to FIFO-pop later
                # and a sub finishing out of completion order lands its
                # result on the wrong chip.
                parent_label = ev.parent_agent_id or "main"
                bound_tc_id = _pop_tc(f"{parent_label}/callagent")
                if bound_tc_id:
                    callagent_chip_by_sub[ev.agent_id] = bound_tc_id
                yield StepStartedEvent(step_name=sub_step)

            elif ev.type == "subagent_end":
                # Resolve the chip bound at subagent_start time so the result
                # content lands on the SAME chip the UI labelled "→ {sub}".
                # Fall back to FIFO /callagent pop only if we never bound
                # (defensive — shouldn't happen in normal flow).
                parent_label = ev.parent_agent_id or "main"
                tc_id = callagent_chip_by_sub.pop(ev.agent_id, None) or _pop_tc(
                    f"{parent_label}/callagent"
                )
                if tc_id:
                    msg_id = _msg_id(parent_label)
                    yield ToolCallResultEvent(
                        message_id=msg_id,
                        tool_call_id=tc_id,
                        content=_cap(ev.content or ""),
                        role="tool",
                    )
                open_subagent_step.pop(ev.agent_id, None)
                yield StepFinishedEvent(step_name=f"{ev.agent_id}/subagent")

            elif ev.type == "final_answer":
                # Final answer signals the end of this agent's round loop —
                # close out its open round step so the STEP_STARTED pairs up.
                step = open_round_step.pop(scope_label, None)
                if step is not None:
                    yield StepFinishedEvent(step_name=step)
                # Emit the answer as a TextMessage only when it is NOT already
                # contained in the last reasoning text for this agent. This
                # handles the force-finish path (synthesized answer is new)
                # without re-emitting the normal-path answer (which was
                # already streamed as part of reasoning).
                text = (ev.content or "").strip()
                if text:
                    prev = last_reasoning_per_agent.get(scope_label, "")
                    if text not in prev:
                        # Open a fresh round so the recovery answer doesn't
                        # land on — and overwrite — the last reasoning round.
                        # Without this, force-finish wipes the model's last
                        # incomplete trace from the UI as a side effect.
                        next_round = last_round_num_per_agent.get(scope_label, 0) + 1
                        recovery_step = f"{scope_label}/round-{next_round}"
                        last_round_num_per_agent[scope_label] = next_round
                        yield StepStartedEvent(step_name=recovery_step)
                        msg_id = _msg_id(scope_label)
                        yield TextMessageStartEvent(message_id=msg_id, role="assistant")
                        yield TextMessageContentEvent(message_id=msg_id, delta=text)
                        yield TextMessageEndEvent(message_id=msg_id)
                        yield StepFinishedEvent(step_name=recovery_step)
                        last_reasoning_per_agent[scope_label] = text
    finally:
        # If we exit via GeneratorExit (FastAPI closed the SSE because the
        # client disconnected), signal the engine to wind down at the next
        # round boundary. Without this, the worker thread keeps spending
        # compute on an answer nobody will read.
        cancel_event.set()
        # Stop the disconnect watcher (it's redundant once cancel_event is
        # set, and must not outlive the run).
        watcher_task.cancel()

    # Close any lingering round steps (e.g. on RUN_ERROR before final_answer)
    # so clients observe a well-formed step-event tree.
    for step in list(open_round_step.values()):
        yield StepFinishedEvent(step_name=step)
    open_round_step.clear()
    # Same for unmatched ``/subagent`` steps — if the engine errored before
    # CallAgentProcessor emitted subagent_end, the sub's StepStarted is
    # still open. Close it here so strict AG-UI clients don't see an
    # unmatched start.
    for step in list(open_subagent_step.values()):
        yield StepFinishedEvent(step_name=step)
    open_subagent_step.clear()

    if error_to_raise is not None:
        yield RunErrorEvent(message=str(error_to_raise))
    else:
        yield RunFinishedEvent(thread_id=thread_id, run_id=run_id)


# ─── HTTP endpoints ───────────────────────────────────────────────────────


_STARTED_AT = __import__("time").time()


@app.get("/health")
def health() -> dict[str, Any]:
    import time as _time

    from athena_r1 import __version__

    payload: dict[str, Any] = {
        "status": "ok",
        "version": __version__,
        "uptime_s": round(_time.time() - _STARTED_AT, 1),
        "initialized": _agent is not None,
    }
    if _agent is not None:
        # Surface the full agent config when ready — useful for monitoring.
        payload["agent"] = _agent.info()
    return payload


# Mount the bundled HTML demo at /demo so users have a zero-install
# way to verify the server works in a browser.
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    # `no-store` ensures the browser fetches the latest HTML on every
    # reload — the demo iterates fast and Etag-based caching makes
    # users wonder why a UI tweak didn't take effect.
    _NO_CACHE = {"Cache-Control": "no-store, must-revalidate"}

    @app.get("/demo")
    def demo() -> FileResponse:
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"), headers=_NO_CACHE)

    @app.get("/")
    def root_get() -> FileResponse:
        """GET / serves the demo; POST / hits the AG-UI endpoint."""
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"), headers=_NO_CACHE)


@app.post("/")
async def agui_endpoint(input_data: RunAgentInput, request: Request) -> StreamingResponse:
    """Standard AG-UI agent endpoint. Accepts a `RunAgentInput` and streams events."""
    accept_header = request.headers.get("accept")
    encoder = EventEncoder(accept=accept_header)

    question = _fold_messages_to_prompt(input_data.messages)
    thread_id = input_data.thread_id or f"thread-{uuid.uuid4().hex[:16]}"
    run_id = input_data.run_id or f"run-{uuid.uuid4().hex[:16]}"

    # Per-request multi-agent toggle. Prefer the explicit header (set by
    # the bundled HTML demo); fall back to forwarded_props.multi_agent if
    # the client only set the AG-UI spec field; default ON otherwise.
    header_val = request.headers.get("x-athena-multi-agent")
    if header_val is not None:
        multi_agent_requested = header_val.strip().lower() in ("1", "true", "yes")
    else:
        # forwarded_props may arrive as a plain dict OR (depending on the
        # ag_ui pydantic model version) as a sub-model / object. Read it
        # tolerantly: dict-style .get first, then attribute access. Only a
        # value that is explicitly present and falsy disables multi-agent.
        fp = getattr(input_data, "forwarded_props", None)
        if isinstance(fp, dict):
            ma = fp.get("multi_agent", None)
        elif fp is not None:
            ma = getattr(fp, "multi_agent", None)
        else:
            ma = None
        multi_agent_requested = True if ma is None else bool(ma)

    async def event_generator() -> AsyncIterator[str]:
        if not question:
            yield encoder.encode(RunStartedEvent(thread_id=thread_id, run_id=run_id))
            yield encoder.encode(RunErrorEvent(message="no user message in input"))
            return
        async for event in _translate_to_agui(
            question,
            thread_id,
            run_id,
            multi_agent_requested=multi_agent_requested,
            request=request,
        ):
            yield encoder.encode(event)

    return StreamingResponse(
        event_generator(),
        media_type=encoder.get_content_type(),
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ReportRequest(BaseModel):
    question: str
    final_answer: str = ""
    events: list[dict] = []


@app.post("/report")
async def report_endpoint(req: ReportRequest, request: Request):
    """Generate a detailed structured report from a captured run trace (SSE)."""
    from fastapi import HTTPException

    from athena_r1._report import digest_from_events

    if not (req.question and req.question.strip()):
        raise HTTPException(status_code=400, detail="question is required")

    agent = get_agent()
    core = agent._core
    budget = core.report_char_budget()
    digest = digest_from_events(req.question, req.final_answer, req.events, budget)

    cancel_event = _threading.Event()

    async def _gen():
        loop = asyncio.get_running_loop()

        async def _watch():
            try:
                while not cancel_event.is_set():
                    if await request.is_disconnected():
                        cancel_event.set()
                        return
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001
                pass

        watcher = loop.create_task(_watch())
        try:
            # Stream progressive snapshots: each is the FULL cleaned report so
            # far (replace semantics), separated by the record-separator char
            # (U+001E). The client splits on it and renders the last complete
            # record, so the report appears to grow live instead of popping in
            # all at once after the full generation.
            it = core.report_snapshots(req.question, digest, cancel_event=cancel_event)
            while True:
                snap = await loop.run_in_executor(None, lambda: next(it, None))
                if snap is None:
                    break
                yield snap + "\x1e"
        finally:
            cancel_event.set()
            watcher.cancel()

    return StreamingResponse(
        _gen(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("AGUI_PORT", "8090")),
    )


if __name__ == "__main__":
    main()
