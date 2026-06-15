"""Public API for ATHENA-R1.

This module exposes the high-level `AthenaR1` class that wraps the internal
multi-step reasoning engine in a clean interface suitable for both interactive
use (web demos, programmatic clients) and structured evaluation (benchmark
pipelines that require letter-letter option mapping).

Design notes:
    * `AthenaR1.answer(question)` performs Stage-1 multi-step tool reasoning
      and returns the model's free-form final answer. No multiple-choice
      options are needed — this is the primary entry point for real-world use.
    * `AthenaR1.map_to_option(...)` is a separate function used only when an
      MCQ-style letter mapping is required (eval, benchmarks). It can be
      driven either by the local ATHENA-R1 model (`backend="athena"`) or by
      Azure GPT-5 (`backend="gpt"`).
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ._core import AthenaCore as _AthenaCore

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*({.*?})\s*</tool_call>", flags=re.DOTALL)


class Backend(str, Enum):
    """LLM backend choice for Stage-1 reasoning or Stage-2 option mapping."""

    ATHENA = "athena"
    GPT = "gpt"


@dataclass
class AnswerResult:
    """Result of `AthenaR1.answer()`.

    Attributes:
        answer: The model's free-form final answer (Stage-1 output).
        conversation: Full multi-turn chat history including tool calls and
            results. Pass this back to `map_to_option()` if you later need an
            MCQ letter.
        rounds_used: Number of reasoning rounds consumed (1..max_round).
        elapsed_s: Wall-clock seconds spent inside `answer()` (set by the
            wrapper; ``0.0`` if produced through another path).
        tools_used: Names of every distinct biomedical tool the agent called
            during Stage-1 (deduped). Excludes the meta-tools ``Tool_RAG``,
            ``CallAgent``, ``Finish`` — those are infrastructure.
        cancelled: ``True`` iff a ``timeout=`` argument fired and the engine
            force-finished from a partial reasoning trace. ``False`` means
            the agent emitted a normal ``[FinalAnswer]``.
        forced: ``True`` iff the engine had to synthesize the answer because
            the run hit its ``max_round`` budget without the model emitting
            a natural ``[FinalAnswer]``. Detected via ``rounds_used >=
            max_round_used``. The OpenAI-compat server maps this to the
            ``length`` finish_reason so SDK clients can distinguish a
            naturally-completed answer from a truncated one. ``cancelled``
            covers the orthogonal case where a wall-clock timeout fired.
    """

    answer: str
    conversation: list[dict[str, Any]] = field(default_factory=list)
    rounds_used: int = 0
    elapsed_s: float = 0.0
    tools_used: list[str] = field(default_factory=list)
    cancelled: bool = False
    forced: bool = False


@dataclass
class RoundEvent:
    """Streaming event emitted during multi-step reasoning.

    Attributes:
        type: One of:
            * ``round_start``: a new reasoning round began
            * ``reasoning``: the model produced reasoning text for this round
            * ``tool_call``: a ``<tool_call>...</tool_call>`` block was parsed
              (may be ``Tool_RAG``, ``CallAgent``, or a regular biomedical tool;
              UI should branch on ``metadata['tool_name']``)
            * ``tool_rag_query``: a ``Tool_RAG`` lookup was issued; ``content`` is the query
            * ``tools_retrieved``: ``Tool_RAG`` returned a tool list;
              ``metadata['tools']`` lists ``{name, description}``
            * ``tool_result``: a tool returned a result and was appended to history
            * ``subagent_start``: a child agent (via ``CallAgent``) is starting;
              ``content`` is the child's sub-question;
              ``agent_id`` is the new child id; ``parent_agent_id`` is the spawner
            * ``subagent_end``: child agent finished; ``content`` is its free-form answer
            * ``final_answer``: the agent emitted its final answer; iteration ends
        content: Primary text payload (event-type-dependent).
        round: Round number this event belongs to (1-indexed) if applicable.
        agent_id: Unique id for the emitting agent. ``"main"`` for the top
            agent; sub-agents get ids like ``"main.sub-1"``, ``"main.sub-1.sub-1"``.
        agent_level: 0 for main agent, +1 per nesting level.
        parent_agent_id: For sub-agents, the id of the spawning agent. ``None``
            for the main agent.
        metadata: Free-form extra fields (tool name, forced-finish flag, ...).
    """

    type: str
    content: str | None = None
    round: int | None = None
    agent_id: str = "main"
    agent_level: int = 0
    parent_agent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AthenaR1:
    """ATHENA-R1: an AI agent for treatment reasoning over a biomedical tool universe.

    Example (web / programmatic use, no options):
        >>> agent = AthenaR1(
        ...     model="mims-harvard/ATHENA-R1-Qwen3-8B",
        ...     vllm_url="http://0.0.0.0:8000/v1",
        ...     tool_server="http://0.0.0.0:8080",
        ... )
        >>> result = agent.answer("How should I dose drug X for a 65yo with CKD?")
        >>> print(result.answer)

    Example (eval / MCQ, with options):
        >>> result = agent.answer(question)
        >>> letter = agent.map_to_option(
        ...     result.conversation,
        ...     options={"A": "...", "B": "...", "C": "...", "D": "..."},
        ...     backend=Backend.ATHENA,
        ... )
    """

    def __init__(
        self,
        model: str | None = None,
        vllm_url: str | None = None,
        tool_server: str | None = None,
        backend: Backend = Backend.ATHENA,
        nested_backend: Backend | None = None,
        azure_endpoint: str | None = None,
        azure_api_key: str | None = None,
        azure_api_version: str = "2024-12-01-preview",
        rag_model: str = "mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B",
        tool_categories: list[str] | None = None,
        max_round: int = 40,
        max_token: int = 36864,
        init_rag_num: int = 0,
        step_rag_num: int = 5,
        presence_penalty: float = 0.0,
        force_finish: bool = True,
        enable_summary: bool = False,
        cache_tool: bool = False,
        max_agent_level: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the agent.

        Args:
            model: HuggingFace model id or local path of the ATHENA-R1 model
                weights. Required unless `backend=Backend.GPT`.
            vllm_url: URL of the vLLM OpenAI-compatible chat completion endpoint
                serving the model. Required unless `backend=Backend.GPT`.
                Example: ``http://0.0.0.0:8000/v1``.
            tool_server: URL of a ToolUniverse HTTP server. Recommended for
                multi-process deployments. If None, ToolUniverse is loaded
                in-process (uses ~15GB RAM).
            backend: Which LLM provides Stage-1 reasoning. ``ATHENA`` (default)
                routes through the local vLLM-served model; ``GPT`` routes
                through Azure OpenAI.
            nested_backend: Backend used by call_agent sub-agents (level >= 1).
                Defaults to the same as `backend`.
            azure_endpoint: Azure OpenAI endpoint URL. Required if `backend=GPT`
                or if you intend to use `map_to_option(backend="gpt")`.
            azure_api_key: Azure OpenAI API key. Falls back to environment
                variable ``AZURE_API_KEY`` if None.
            azure_api_version: Azure OpenAI API version string.
            rag_model: HuggingFace id of the ToolRAG retrieval model.
            tool_categories: List of ToolUniverse categories to load. Defaults
                to the paper-canonical set (tool_finder, opentarget,
                fda_drug_label, special_tools, monarch, fda_drug_adverse_event,
                ChEMBL, EuropePMC, semantic_scholar, pubtator, EFO).
            max_round: Maximum multi-step reasoning rounds before force-finish.
            max_token: Maximum input tokens fed to vLLM per call.
            init_rag_num: Number of tools pre-loaded into the prompt at start
                of each question (paper default 0).
            step_rag_num: Number of tools Tool_RAG can retrieve per request
                (paper default 5).
            presence_penalty: Qwen3 sampling parameter. Paper-canonical = 0.0;
                values 0–2 reduce repetition at a small accuracy cost.
            force_finish: If True, synthesize a final answer when `max_round`
                is reached. If False, return None on exhaustion.
            enable_summary: Enable in-loop tool-output summarization (advanced).
            cache_tool: If True, reuse cached tool outputs across questions
                (paper default False for honest eval).
            **kwargs: Passed through to the underlying engine.

        Environment variables (used as fallbacks):
            ``TOOLUNIVERSE_API``  same as ``tool_server`` arg
            ``AZURE_API_KEY``     same as ``azure_api_key`` arg
            ``AZURE_OPENAI_ENDPOINT``  same as ``azure_endpoint`` arg
        """
        if tool_server is not None:
            os.environ["TOOLUNIVERSE_API"] = tool_server

        # Validate caller-supplied required combinations early so misuse
        # surfaces at construction time, not at first request time.
        if backend == Backend.ATHENA and not (model and vllm_url):
            raise ValueError(
                "Backend.ATHENA requires both `model` and `vllm_url`. "
                "Pass them, or use `backend=Backend.GPT` if you only want "
                "to call Azure GPT-5."
            )
        if backend == Backend.GPT and not (
            azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        ):
            raise ValueError(
                "Backend.GPT requires `azure_endpoint=...` (or the AZURE_OPENAI_ENDPOINT env var)."
            )
        if max_round <= 0:
            raise ValueError(f"max_round must be > 0, got {max_round}")
        if not 0.0 <= presence_penalty <= 2.0:
            raise ValueError(f"presence_penalty must be in [0, 2], got {presence_penalty}")

        # Azure config is consumed by the underlying engine through env vars.
        if azure_endpoint is not None:
            os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
        if azure_api_key is not None:
            os.environ["AZURE_API_KEY"] = azure_api_key
        os.environ.setdefault("AZURE_OPENAI_API_VERSION", azure_api_version)

        # `presence_penalty` is consumed inside the core engine via the
        # INFER_PRESENCE_PENALTY env var (see _core.llm_infer).
        # 0.0 is the paper default ("no presence penalty").
        os.environ["INFER_PRESENCE_PENALTY"] = str(float(presence_penalty))

        backend_default = backend.value
        backend_by_level: dict[int, str] = {0: backend_default}
        if nested_backend is not None:
            backend_by_level[1] = nested_backend.value

        # Canonical paper-matching set of tool categories. Override via
        # `tool_categories` if you want a different subset.
        if tool_categories is None:
            tool_categories = [
                "tool_finder",
                "opentarget",
                "fda_drug_label",
                "special_tools",
                "monarch",
                "fda_drug_adverse_event",
                "ChEMBL",
                "EuropePMC",
                "semantic_scholar",
                "pubtator",
                "EFO",
            ]

        self._core = _AthenaCore(
            model_name=model,
            rag_model_name=rag_model,
            vllm_server_url=vllm_url or "",
            enable_summary=enable_summary,
            force_finish=force_finish,
            cache_tool=cache_tool,
            init_rag_num=init_rag_num,
            step_rag_num=step_rag_num,
            add_call_id=False,
            tool_in_results=True,
            max_token=max_token,
            backend_default=backend_default,
            backend_by_level=backend_by_level,
            **kwargs,
        )
        # Override hardcoded depth in _core: when max_agent_level > 0,
        # CallAgent recursion is enabled.
        self._core.max_agent_level = max_agent_level
        self._tool_categories = tool_categories
        self._max_round = max_round
        self._initialized = False

    def init(self) -> None:
        """Load the model + ToolUniverse. Idempotent."""
        if self._initialized:
            return
        self._core.init_model(tool_type=list(self._tool_categories))
        self._initialized = True

    def _run_with_timeout(
        self,
        prompt: str,
        timeout: float,
        **engine_kwargs: Any,
    ) -> tuple[str | None, list[dict[str, Any]], bool]:
        """Drive run_multistep_agent in a worker thread with a wall-clock deadline.

        Uses a per-call ``threading.Event`` to signal cancellation, so two
        concurrent ``answer()`` invocations on the same agent instance don't
        interfere with each other. Whatever conversation has accumulated up to
        the deadline is returned. Third tuple element is True iff the
        deadline fired (the agent was force-finished from a partial trace).
        """
        result_holder: dict[str, Any] = {}
        cancel_event = threading.Event()

        def runner() -> None:
            try:
                resp, conv = self._core.run_multistep_agent(
                    prompt,
                    return_conversation=True,
                    cancel_event=cancel_event,
                    **engine_kwargs,
                )
                result_holder["resp"] = resp
                result_holder["conv"] = conv
            except Exception as e:  # noqa: BLE001
                result_holder["error"] = e

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        cancelled = False
        if thread.is_alive():
            # Timed out — ask the engine to wind down.
            cancel_event.set()
            cancelled = True
            # Give it a short grace period to emit force_finish.
            thread.join(timeout=30.0)
        if "error" in result_holder:
            raise result_holder["error"]
        return (
            result_holder.get("resp"),
            result_holder.get("conv", []),
            cancelled,
        )

    # Paper-canonical Stage-1 prompt template. Keeps the model's reasoning
    # format consistent with the format it was trained on; deviating from
    # this template reduces accuracy.
    _STAGE1_PROMPT_TEMPLATE = (
        "The following are open-ended questions about medical. "
        "In the final answer, give a comprehensive answer and explain the reason. "
        "The question is: \n{question}\n\nThe answer is:"
    )

    def answer(
        self,
        question: str,
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_round: int | None = None,
        max_new_tokens: int = 4096,
        call_agent: bool = False,
        wrap_prompt: bool = True,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> AnswerResult:
        """Run Stage-1 multi-step tool reasoning and return a free-form answer.

        This is the primary inference entry point. No multiple-choice options
        are needed; the model returns natural-language reasoning + a final
        answer. For MCQ letter mapping, call `map_to_option()` afterwards.

        Args:
            question: The user's clinical/biomedical question.
            temperature, top_p, top_k, min_p: Stage-1 sampling parameters.
                Recommended (matches paper): T=0.7, top_p=0.95, top_k=20.
            max_round: Override the instance-level ``max_round``.
            max_new_tokens: Per-call token budget for vLLM.
            call_agent: Enable recursive sub-agent calls (advanced).
            wrap_prompt: If True (default), wrap `question` in the paper-
                canonical "The following are open-ended questions ..."
                template. Required for full paper-grade accuracy. Set to
                False if you want to control the prompt yourself.
            timeout: Optional wall-clock timeout in seconds. If reached, the
                worker is asked to stop at the next round boundary; whatever
                partial answer has been emitted so far is returned. ``None``
                (default) means no timeout (matches paper).
            **kwargs: Forwarded to the engine.

        Returns:
            AnswerResult with `answer` (free-form text), `conversation` (full
            chat history including tool calls), and `rounds_used`.
        """
        if not self._initialized:
            self.init()

        if wrap_prompt:
            prompt = self._STAGE1_PROMPT_TEMPLATE.format(question=question)
        else:
            prompt = question

        _t0 = time.monotonic()
        cancelled = False
        if timeout is None:
            # Fast path: synchronous call with no timeout wrapper.
            response, conversation = self._core.run_multistep_agent(
                prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_round=max_round or self._max_round,
                max_new_tokens=max_new_tokens,
                call_agent=call_agent,
                return_conversation=True,
                **kwargs,
            )
        else:
            # Timeout path: drive run_multistep_agent in a thread and join
            # with a deadline. If we hit the deadline, we set a cancel flag
            # the engine polls at every round boundary.
            response, conversation, cancelled = self._run_with_timeout(
                prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_round=max_round or self._max_round,
                max_new_tokens=max_new_tokens,
                call_agent=call_agent,
                timeout=timeout,
                **kwargs,
            )
        elapsed_s = round(time.monotonic() - _t0, 2)

        if response is None:
            response = ""

        rounds_used = sum(
            1 for msg in conversation if isinstance(msg, dict) and msg.get("role") == "assistant"
        )
        tools_used = self._extract_tools_used(conversation)
        # When the model fails to emit ``[FinalAnswer]`` within ``max_round``,
        # the engine synthesizes an answer via
        # ``get_answer_based_on_unfinished_reasoning``. The conversation then
        # has exactly ``max_round`` assistant messages (one per round). A
        # natural finish terminates EARLY, so a strict equal-or-greater
        # comparison cleanly distinguishes the two paths. ``cancelled``
        # already captures the orthogonal wall-clock timeout case.
        max_round_used = max_round or self._max_round
        forced = (not cancelled) and rounds_used >= max_round_used

        return AnswerResult(
            answer=response,
            conversation=conversation,
            rounds_used=rounds_used,
            elapsed_s=elapsed_s,
            tools_used=tools_used,
            forced=forced,
            cancelled=cancelled,
        )

    def generate_report(
        self,
        result_or_conversation,
        *,
        stream: bool = False,
        temperature: float = 0.7,
        max_new_tokens: int = 4096,
    ):
        """Generate a detailed structured clinical report from a completed run.

        Args:
            result_or_conversation: An ``AnswerResult`` from ``answer()`` or a
                raw conversation list.
            stream: If True, return an iterator of text chunks; otherwise return
                the full report string.
        """
        from athena_r1._report import digest_from_conversation

        conversation = getattr(result_or_conversation, "conversation", result_or_conversation)
        question = ""
        for m in conversation or []:
            if isinstance(m, dict) and m.get("role") == "user":
                question = str(m.get("content", "") or "")
                break
        budget = self._core.report_char_budget(max_new_tokens)
        digest = digest_from_conversation(conversation, budget)
        gen = self._core.generate_report(
            question, digest, temperature=temperature, max_new_tokens=max_new_tokens
        )
        if stream:
            return gen
        return "".join(gen)

    def answer_streaming(
        self,
        question: str,
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_round: int | None = None,
        max_new_tokens: int = 4096,
        call_agent: bool = False,
        wrap_prompt: bool = True,
        timeout: float | None = None,
        external_cancel_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> Iterator[RoundEvent]:
        """Stream Stage-1 reasoning events live as they happen.

        Yields a sequence of ``RoundEvent`` objects matching the agent's
        progress through multi-step tool reasoning. The generator finishes
        either when the agent emits ``final_answer`` or when the underlying
        thread completes (e.g. forced finish at max_round).

        Each event type emits a non-overlapping payload:
            * ``round_start``: only ``round`` + ``metadata["max_round"]``
            * ``reasoning``: ``content`` = full assistant message for the round
              (includes any ``<tool_call>`` blocks inline)
            * ``tool_call``: ``content`` = raw JSON inside the ``<tool_call>`` block
            * ``tool_result``: ``content`` = tool output text, ``metadata["name"]``
              = tool name
            * ``final_answer``: ``content`` = final natural-language answer

        After ``final_answer`` is emitted, the generator stops. Use the
        ``conversation`` field on the *parent* iterator (via
        ``RoundEvent.metadata['conversation']`` on the final event) if you
        need it for ``map_to_option`` downstream.

        Example:
            >>> for ev in agent.answer_streaming("..."):
            ...     if ev.type == "tool_call":
            ...         print(f"calling {ev.content}")
            ...     elif ev.type == "final_answer":
            ...         print(f"\\nFinal: {ev.content}")
        """
        if not self._initialized:
            self.init()

        if wrap_prompt:
            prompt = self._STAGE1_PROMPT_TEMPLATE.format(question=question)
        else:
            prompt = question

        event_queue: queue.Queue = queue.Queue()
        SENTINEL_DONE = {"_done": True}
        result_holder: dict[str, Any] = {}
        # Per-call cancel event — independent of any concurrent answer()
        # call on the same agent instance. Callers (e.g. the AG-UI server)
        # can pass their own ``external_cancel_event`` to wire client-side
        # cancellation (SSE client disconnect) all the way down to the
        # engine's round-boundary cancellation poll.
        cancel_event = external_cancel_event or threading.Event()
        deadline = None if timeout is None else (time.monotonic() + timeout)

        def push(event: dict[str, Any]) -> None:
            event_queue.put(event)

        def runner() -> None:
            try:
                response, conversation = self._core.run_multistep_agent(
                    prompt,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_round=max_round or self._max_round,
                    max_new_tokens=max_new_tokens,
                    call_agent=call_agent,
                    return_conversation=True,
                    progress_callback=push,
                    cancel_event=cancel_event,
                    **kwargs,
                )
                result_holder["response"] = response or ""
                result_holder["conversation"] = conversation
            except Exception as e:  # noqa: BLE001
                # Store the error FIRST so even if the traceback logging
                # below were to fail, the caller still sees the error.
                result_holder["error"] = e
                # Log the full traceback to stderr so error events on the SSE
                # side carry enough context to diagnose — without it the
                # client sees only `str(e)`. Use sys.stderr directly (this
                # module doesn't import a logger).
                try:
                    import sys as _sys
                    import traceback as _tb

                    print(
                        "answer_streaming runner crashed:\n" + _tb.format_exc(),
                        file=_sys.stderr,
                        flush=True,
                    )
                except Exception:  # noqa: BLE001
                    pass
            finally:
                event_queue.put(SENTINEL_DONE)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()

        try:
            while True:
                try:
                    # Honor caller timeout cooperatively.
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            cancel_event.set()
                            event = event_queue.get(timeout=30.0)
                        else:
                            event = event_queue.get(timeout=remaining)
                    else:
                        event = event_queue.get()
                except queue.Empty:
                    # Timeout hit waiting for the next event — request stop.
                    cancel_event.set()
                    continue
                if event is SENTINEL_DONE:
                    thread.join()
                    err = result_holder.get("error")
                    if err is not None:
                        raise err
                    return
                # Convert dict → RoundEvent dataclass for nicer ergonomics.
                etype = event.pop("type")
                content = event.pop("content", None)
                round_n = event.pop("round", None)
                agent_id = event.pop("agent_id", "main")
                agent_level = event.pop("agent_level", 0)
                parent_id = event.pop("parent_agent_id", None)
                yield RoundEvent(
                    type=etype,
                    content=content,
                    round=round_n,
                    agent_id=agent_id,
                    agent_level=agent_level,
                    parent_agent_id=parent_id,
                    metadata=event,
                )
                # Only the MAIN agent's final_answer signals end-of-run. The
                # engine emits a `final_answer` for every sub-agent too as it
                # finishes its recursive run_multistep_agent — returning on
                # that one drops every subsequent main-agent event (subagent_end,
                # more rounds, the actual top-level final_answer). Gate on
                # ``agent_level == 0`` so we only short-circuit when the
                # top-level run has actually finished.
                if etype == "final_answer" and agent_level == 0:
                    # Drain the queue, then exit so callers don't have to filter.
                    thread.join(timeout=2.0)
                    return
        finally:
            # Caller broke out of the generator early (e.g. user clicked Stop
            # in a UI). Signal *this call's* worker to wind down at the next
            # round boundary so we don't leak compute on a runaway agent.
            cancel_event.set()

    def map_to_option(
        self,
        conversation: list[dict[str, Any]],
        options: dict[str, str],
        *,
        backend: Backend = Backend.ATHENA,
        temperature: float = 0.1,
        max_new_tokens: int = 1024,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        question: str | None = None,
    ) -> str:
        """Extract a single option letter (e.g. "A", "B") for an MCQ task.

        This is Stage-2 of the eval pipeline: given the full conversation
        produced by `answer()` and a set of option letters → texts, map the
        model's free-form answer to one of the letters.

        Two backends are available:
            * ``Backend.ATHENA``: re-uses the local model via the model's
              own self-extraction prompt format.
            * ``Backend.GPT``: uses Azure GPT-5 to read the conversation and
              pick a letter. More robust to phrasing variation.

        Args:
            conversation: The chat history returned by `answer()`.
            options: Letter → option text mapping, e.g.
                ``{"A": "Aspirin 81mg", "B": "Clopidogrel", "C": ...}``.
            backend: Which LLM performs the mapping.
            temperature, top_p, top_k, min_p, max_new_tokens: Stage-2 sampling
                parameters (only used when ``backend=ATHENA``).
            question: Original question text. Required only when
                ``backend=GPT``.

        Returns:
            A single uppercase letter ("A", "B", ...) or the literal string
            "Error" if extraction failed.
        """
        if not self._initialized:
            self.init()

        options_str = ", ".join(f"{k}. {v}" for k, v in options.items())

        if backend == Backend.ATHENA:
            return self._core.run_format_agent_full_history(
                options_str=options_str,
                conversation_history=conversation,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
            )

        if backend == Backend.GPT:
            if question is None:
                question = self._extract_question_from_conversation(conversation)
            return self._core.run_gpt_format_agent(
                conversation_history=conversation,
                options_str=options_str,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                message=question,
            )

        raise ValueError(f"Unknown backend: {backend}")

    # Meta-tools that are infrastructure rather than knowledge sources;
    # we hide them from `tools_used` so eval scripts don't double-count.
    _META_TOOLS = frozenset(
        {"Tool_RAG", "CallAgent", "Finish", "DirectResponse", "RequireClarification"}
    )

    @classmethod
    def _extract_tools_used(cls, conversation: list[dict[str, Any]]) -> list[str]:
        """Deduped list of biomedical tool names invoked during Stage-1."""
        seen: list[str] = []
        for msg in conversation:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            content = msg.get("content") or ""
            for m in _TOOL_CALL_RE.finditer(content):
                try:
                    name = json.loads(m.group(1)).get("name")
                except Exception:  # noqa: BLE001
                    name = None
                if name and name not in cls._META_TOOLS and name not in seen:
                    seen.append(name)
        return seen

    @staticmethod
    def _extract_question_from_conversation(conv: list[dict[str, Any]]) -> str:
        for msg in conv:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
        return ""

    # ─── Lifecycle & introspection ─────────────────────────────────────────

    def close(self) -> None:
        """Mark the agent as closed and release lazily-held clients.

        The vLLM and ToolUniverse clients are HTTP wrappers and don't hold
        persistent sockets, so they're allowed to lapse to garbage collection.
        Azure-OpenAI client is dropped explicitly so a subsequent ``re-init``
        with different credentials picks up the new env vars.
        """
        self._initialized = False
        # Drop the lazy Azure client so a re-init can pick up new credentials.
        if hasattr(self._core, "_gpt_client_obj"):
            self._core._gpt_client_obj = None

    def __enter__(self) -> AthenaR1:
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        model = getattr(self._core, "model_name", "?")
        vllm = getattr(self._core, "vllm_server_url", "?")
        ready = "ready" if self._initialized else "not-initialized"
        return (
            f"AthenaR1(model={model!r}, vllm={vllm!r}, max_round={self._max_round}, status={ready})"
        )

    def info(self) -> dict[str, Any]:
        """JSON-serializable summary of the agent's current configuration.

        Useful for monitoring endpoints, audit logs, and ``/health`` payloads.
        Reading this does NOT require the agent to be initialized.
        """
        c = self._core
        tu = getattr(c, "tooluniverse", None)
        try:
            n_loaded_tools = len(tu.return_all_loaded_tools()) if tu else 0
        except Exception:  # noqa: BLE001
            n_loaded_tools = 0
        return {
            "model": getattr(c, "model_name", None),
            "vllm_url": getattr(c, "vllm_server_url", None),
            "tool_server": os.environ.get("TOOLUNIVERSE_API"),
            "backend": getattr(c, "backend_default", "athena"),
            "max_round": self._max_round,
            "max_token": getattr(c, "max_token", None),
            "max_agent_level": getattr(c, "max_agent_level", 0),
            "init_rag_num": getattr(c, "init_rag_num", None),
            "step_rag_num": getattr(c, "step_rag_num", None),
            "cache_tool": getattr(c, "cache_tool", None),
            "enable_rag": getattr(c, "enable_rag", None),
            "tool_categories": list(self._tool_categories),
            "tools_loaded": n_loaded_tools,
            "initialized": self._initialized,
        }
